from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "bundle_tool.py"
SPEC = importlib.util.spec_from_file_location(
    "splatlab_unreal_bundle_tool", MODULE_PATH
)
assert SPEC and SPEC.loader
bundle_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle_tool
SPEC.loader.exec_module(bundle_tool)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(root: Path, *, calibrated: bool = True) -> Path:
    gaussian = root / "Gaussian" / "scene.ply"
    gaussian.parent.mkdir(parents=True)
    gaussian.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 2 3",
                "",
            ]
        ),
        encoding="ascii",
    )
    meters = 0.01 if calibrated else None
    manifest = {
        "schema": bundle_tool.BUNDLE_SCHEMA,
        "job_id": "unit-job_01",
        "created_at": "2026-07-24T00:00:00+00:00",
        "target": {
            "engine": "Unreal Engine",
            "version": "5.6",
            "platform": "Windows",
            "renderer_probe_order": [
                "NanoGS",
                "MLSLabsRenderer",
                "UnrealSplat",
            ],
        },
        "coordinate_system": {
            "source_frame": {
                "handedness": "right-handed",
                "right_axis": "+X",
                "forward_axis": "+Y",
                "up_axis": "+Z",
                "units": "scene-units",
                "meters_per_unit": meters,
                "scale_calibrated": calibrated,
            },
            "georeference": None,
        },
        "import_contract": {
            "source_up_axis": "+Z",
            "source_forward_axis": "+Y",
            "target_up_axis": "+Z",
            "target_forward_axis": "+X",
            "target_units": "centimeters",
            "centimeters_per_source_unit": 1.0 if calibrated else None,
            "requires_operator_alignment": not calibrated,
            "actor_structure": {
                "root": "SplatLabSceneRoot",
                "children": [
                    "GaussianRender",
                    "Collision",
                    "ConventionalGeometry",
                ],
            },
        },
        "files": [
            {
                "path": "Gaussian/scene.ply",
                "role": "canonical-gaussian-splat",
                "bytes": gaussian.stat().st_size,
                "sha256": _sha256(gaussian),
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    return root


def _make_project(root: Path, renderer: str | None = None) -> Path:
    root.mkdir()
    (root / "SplatLabUE56.uproject").write_text(
        json.dumps({"FileVersion": 3, "EngineAssociation": "5.6"}),
        encoding="utf-8",
    )
    if renderer:
        descriptor = root / "Plugins" / renderer / f"{renderer}.uplugin"
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text(
            json.dumps(
                {
                    "FileVersion": 3,
                    "FriendlyName": renderer,
                    "Version": 7,
                    "VersionName": "test",
                    "EngineVersion": "5.6.0",
                }
            ),
            encoding="utf-8",
        )
    return root


def test_verify_bundle_checks_contract_files_and_ply_count(tmp_path: Path) -> None:
    verified = bundle_tool.verify_bundle(_make_bundle(tmp_path))

    assert verified.gaussian_count == 2
    assert verified.verified_files == 1
    assert verified.total_bytes > 0
    assert verified.summary()["status"] == "verified"


def test_verify_bundle_supports_explicit_uncalibrated_contract(tmp_path: Path) -> None:
    verified = bundle_tool.verify_bundle(_make_bundle(tmp_path, calibrated=False))

    assert verified.manifest["import_contract"]["requires_operator_alignment"] is True


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda manifest: manifest.update(schema="unknown"), "schema"),
        (lambda manifest: manifest["target"].update(version="5.7"), "version"),
        (
            lambda manifest: manifest["files"][0].update(path="../scene.ply"),
            "Unsafe bundle path",
        ),
        (
            lambda manifest: manifest["files"][0].update(path="Gaussian/CON.ply"),
            "unsafe on Windows",
        ),
        (
            lambda manifest: manifest["files"][0].update(sha256="0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda manifest: manifest["import_contract"].update(
                centimeters_per_source_unit=2.0
            ),
            "does not match meters_per_unit",
        ),
        (
            lambda manifest: manifest["coordinate_system"]["source_frame"].update(
                handedness="left-handed"
            ),
            "source frame",
        ),
    ],
)
def test_verify_bundle_rejects_invalid_content(
    tmp_path: Path, mutation, message: str
) -> None:
    root = _make_bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(bundle_tool.BundleToolError, match=message):
        bundle_tool.verify_bundle(root)


def test_materialized_bundle_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "no")

    with pytest.raises(bundle_tool.BundleToolError, match="Unsafe archive path"):
        with bundle_tool.materialized_bundle(archive):
            pass


def test_materialized_bundle_verifies_safe_zip(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path / "source")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in source.rglob("*"):
            if path.is_file():
                output.write(path, Path("handoff") / path.relative_to(source))

    with bundle_tool.materialized_bundle(archive) as root:
        verified = bundle_tool.verify_bundle(root)

    assert verified.gaussian_count == 2


def test_verify_bundle_rejects_symlinked_file_ancestor(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path)
    real_gaussian = root / "RealGaussian"
    (root / "Gaussian").rename(real_gaussian)
    (root / "Gaussian").symlink_to(real_gaussian.name, target_is_directory=True)

    with pytest.raises(bundle_tool.BundleToolError, match="contains a symlink"):
        bundle_tool.verify_bundle(root)


def test_stage_bundle_is_checksum_addressed_and_idempotent(tmp_path: Path) -> None:
    bundle = bundle_tool.verify_bundle(_make_bundle(tmp_path / "bundle"))
    project = _make_project(tmp_path / "project", renderer="NanoGS")

    first = bundle_tool.stage_bundle(bundle, project, [], "auto")
    second = bundle_tool.stage_bundle(bundle, project, [], "auto")

    assert first["status"] == "staged"
    assert first["renderer"] == "NanoGS"
    assert second["status"] == "cached"
    destination = Path(first["staged_path"])
    assert (destination / "manifest.json").is_file()
    assert not (destination / "staging.json").exists()
    pointer = json.loads(
        (project / "SplatLabImports" / "unit-job_01" / "current.json").read_text()
    )
    assert pointer["manifest_sha256"] == bundle.manifest_sha256


def test_stage_bundle_requires_installed_explicit_renderer(tmp_path: Path) -> None:
    bundle = bundle_tool.verify_bundle(_make_bundle(tmp_path / "bundle"))
    project = _make_project(tmp_path / "project")

    with pytest.raises(bundle_tool.BundleToolError, match="not installed"):
        bundle_tool.stage_bundle(bundle, project, [], "NanoGS")


def test_stage_bundle_rejects_symlinked_import_root(tmp_path: Path) -> None:
    bundle = bundle_tool.verify_bundle(_make_bundle(tmp_path / "bundle"))
    project = _make_project(tmp_path / "project", renderer="NanoGS")
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "SplatLabImports").symlink_to(outside, target_is_directory=True)

    with pytest.raises(bundle_tool.BundleToolError, match="contains a symlink"):
        bundle_tool.stage_bundle(bundle, project, [], "auto")

    assert not list(outside.iterdir())


def test_renderer_probe_follows_declared_order(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "project", renderer="MLSLabsRenderer")
    nano = project / "Plugins" / "NanoGS" / "NanoGS.uplugin"
    nano.parent.mkdir(parents=True)
    nano.write_text(json.dumps({"FileVersion": 3}), encoding="utf-8")

    result = bundle_tool.probe_renderers(project, [])

    assert result["selected"] == "NanoGS"
    assert [item["id"] for item in result["renderers"][:2]] == [
        "NanoGS",
        "MLSLabsRenderer",
    ]


def test_checked_in_mcp_policy_is_closed_world() -> None:
    policy = MODULE_PATH.with_name("mcp-policy.json")

    result = bundle_tool.validate_mcp_policy(policy)

    assert result["status"] == "valid"
    assert result["enforcement"] == "contract-only"
    assert "system_control" in result["denied_tools"]
