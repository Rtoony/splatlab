from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "splatlab_research_benchmark", MODULE_PATH
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)
CATALOG = json.loads(MODULE_PATH.with_name("sources.json").read_text())
ADAPTERS = json.loads(MODULE_PATH.with_name("adapters.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(
    *,
    experiment_id: str = "baseline",
    adapter_id: str = "gsplat",
    scope: str = "production-evaluation",
    psnr: float = 28.0,
) -> dict:
    repository = next(
        item["repository_url"]
        for item in CATALOG["candidates"]
        if item["id"] == adapter_id
    )
    return {
        "schema": benchmark.RECEIPT_SCHEMA,
        "experiment_id": experiment_id,
        "created_at": "2026-07-24T12:00:00+00:00",
        "adapter_id": adapter_id,
        "scope": scope,
        "dataset": {
            "id": "small-room-v1",
            "input_sha256": "1" * 64,
            "capture_type": "posed-images",
        },
        "implementation": {
            "kind": "upstream",
            "repository": repository,
            "revision": "2" * 40,
            "dirty": False,
        },
        "environment": {
            "os": "Linux",
            "python": "3.13",
            "gpu": "test-gpu",
            "vram_mb": 12_288,
            "dependencies_sha256": "3" * 64,
        },
        "configuration": {"strategy": "default"},
        "metrics": {
            "wall_seconds": 60.0,
            "peak_ram_mb": 2048.0,
            "peak_vram_mb": 4096.0,
            "output": {"gaussian_count": 100_000, "bytes": 12_000_000},
            "quality": {"psnr_db": psnr, "ssim": 0.91},
        },
        "artifacts": [
            {
                "role": "canonical-ply",
                "path": "outputs/scene.ply",
                "sha256": "4" * 64,
                "bytes": 12_000_000,
            }
        ],
        "outcome": {"status": "pass", "notes": "fixture"},
    }


def test_checked_in_catalog_passes_license_policy() -> None:
    result = benchmark.validate_catalog(CATALOG)

    assert result["status"] == "valid"
    assert "gsplat" in result["p0"]
    assert "splatreg" in result["p0"]


def test_catalog_blocks_restricted_code_from_production() -> None:
    catalog = copy.deepcopy(CATALOG)
    mvs = next(item for item in catalog["candidates"] if item["id"] == "mvsanywhere")
    mvs["integration_mode"] = "production-permissive"

    with pytest.raises(
        benchmark.ResearchValidationError, match="production integration"
    ):
        benchmark.validate_catalog(catalog)


def test_catalog_rejects_integer_license_booleans() -> None:
    catalog = copy.deepcopy(CATALOG)
    catalog["candidates"][0]["license"]["commercial_use"] = 1

    with pytest.raises(benchmark.ResearchValidationError, match="license assessment"):
        benchmark.validate_catalog(catalog)


def test_catalog_rejects_unknown_code_status() -> None:
    catalog = copy.deepcopy(CATALOG)
    catalog["candidates"][0]["code_status"] = "maybe"

    with pytest.raises(benchmark.ResearchValidationError, match="code status"):
        benchmark.validate_catalog(catalog)


def test_adapter_registry_is_disabled_isolated_and_compute_gated() -> None:
    result = benchmark.validate_adapter_registry(ADAPTERS, CATALOG)

    assert result["status"] == "valid"
    assert result["restricted"] == 3
    assert result["gpu"] == 5


def test_adapter_registry_cannot_embed_an_executable_command() -> None:
    adapters = copy.deepcopy(ADAPTERS)
    adapters["adapters"][0]["command"] = ["python", "run.py"]

    with pytest.raises(benchmark.ResearchValidationError, match="executable commands"):
        benchmark.validate_adapter_registry(adapters, CATALOG)


def test_adapter_probe_rejects_checkout_inside_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ADAPTERS["adapters"][0]
    monkeypatch.setenv(adapter["root_env_var"], str(benchmark.REPO_ROOT))

    result = benchmark.probe_adapter_registry(ADAPTERS, CATALOG)
    probe = next(item for item in result["probes"] if item["id"] == adapter["id"])

    assert probe["ready"] is False
    assert probe["policy_violation"] == "checkout-is-inside-splatlab-repository"


def test_receipt_validation_enforces_reproducibility_fields() -> None:
    result = benchmark.validate_receipt(_receipt(), CATALOG)

    assert result["status"] == "valid"
    assert result["quality_metrics"] == ["psnr_db", "ssim"]


def test_failed_receipt_can_record_no_quality_with_notes() -> None:
    receipt = _receipt()
    receipt["outcome"] = {"status": "fail", "notes": "optimizer diverged"}
    receipt["metrics"]["quality"] = {}

    result = benchmark.validate_receipt(receipt, CATALOG)

    assert result["quality_metrics"] == []
    assert result["outcome"] == "fail"


def test_passing_receipt_requires_quality_metrics() -> None:
    receipt = _receipt()
    receipt["metrics"]["quality"] = {}

    with pytest.raises(benchmark.ResearchValidationError, match="Passing benchmarks"):
        benchmark.validate_receipt(receipt, CATALOG)


def test_restricted_adapter_receipt_must_be_research_only() -> None:
    receipt = _receipt(adapter_id="mvsanywhere")

    with pytest.raises(benchmark.ResearchValidationError, match="research-only"):
        benchmark.validate_receipt(receipt, CATALOG)

    receipt["scope"] = "isolated-research"
    assert benchmark.validate_receipt(receipt, CATALOG)["status"] == "valid"


def test_compare_receipts_applies_direction_and_tolerance() -> None:
    baseline = _receipt(psnr=30.0)
    candidate = _receipt(experiment_id="candidate", psnr=29.8)

    accepted = benchmark.compare_receipts(
        baseline,
        candidate,
        CATALOG,
        "metrics.quality.psnr_db",
        "higher",
        1.0,
    )
    rejected = benchmark.compare_receipts(
        baseline,
        candidate,
        CATALOG,
        "metrics.quality.psnr_db",
        "higher",
        0.1,
    )

    assert accepted["status"] == "pass"
    assert rejected["status"] == "regression"


def test_compare_receipts_uses_magnitude_for_negative_metric_tolerance() -> None:
    baseline = _receipt(psnr=-10.0)
    candidate = _receipt(experiment_id="candidate", psnr=-10.5)

    result = benchmark.compare_receipts(
        baseline,
        candidate,
        CATALOG,
        "metrics.quality.psnr_db",
        "higher",
        10.0,
    )

    assert result["status"] == "pass"
    assert result["threshold"] == pytest.approx(-11.0)


def test_export_report_verifies_sizes_ratios_and_files(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    export_root = job_root / "_exports"
    export_root.mkdir(parents=True)
    source = job_root / "_preview" / "scene.ply"
    source.parent.mkdir()
    source.write_bytes(b"canonical-ply")
    spz = export_root / "scene.spz"
    spz.write_bytes(b"spz")
    manifest = {
        "schema": benchmark.EXPORT_SCHEMA,
        "job_id": "fixture-job",
        "source": {
            "path": "_preview/scene.ply",
            "bytes": source.stat().st_size,
            "sha256": _sha(source),
            "gaussian_count": 5,
        },
        "artifacts": {
            "spz": {
                "status": "ready",
                "parameters": {"spz_version": 4},
                "files": [
                    {
                        "path": "scene.spz",
                        "bytes": spz.stat().st_size,
                        "sha256": _sha(spz),
                    }
                ],
            },
            "streamed-sog": {
                "status": "skipped",
                "reason": "small fixture",
            },
        },
    }
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = benchmark.export_report(manifest_path, verify_files=True)

    assert result["checked_files"] == 2
    assert result["source"]["gaussian_count"] == 5
    assert result["artifacts"][0]["name"] == "spz"
    assert result["artifacts"][0]["source_ratio"] == pytest.approx(
        spz.stat().st_size / source.stat().st_size
    )


def test_export_report_rejects_tampered_artifact(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    export_root = job_root / "_exports"
    export_root.mkdir(parents=True)
    source = job_root / "source.ply"
    source.write_bytes(b"source")
    artifact = export_root / "scene.spz"
    artifact.write_bytes(b"artifact")
    manifest = {
        "schema": benchmark.EXPORT_SCHEMA,
        "job_id": "fixture-job",
        "source": {
            "path": "source.ply",
            "bytes": source.stat().st_size,
            "sha256": _sha(source),
            "gaussian_count": 1,
        },
        "artifacts": {
            "spz": {
                "status": "ready",
                "files": [
                    {
                        "path": "scene.spz",
                        "bytes": artifact.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        },
    }
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(benchmark.ResearchValidationError, match="identity mismatch"):
        benchmark.export_report(manifest_path, verify_files=True)
