from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "capability_probe.py"
SPEC = importlib.util.spec_from_file_location("splatlab_capability_probe", MODULE_PATH)
assert SPEC and SPEC.loader
capabilities = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capabilities
SPEC.loader.exec_module(capabilities)


def _make_distribution(site: Path, name: str, version: str) -> None:
    metadata = site / f"{name}-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_static_probe_detects_available_but_unintegrated_strategies(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "env"
    site = prefix / "lib" / "python3.11" / "site-packages"
    gsplat = site / "gsplat"
    strategy = gsplat / "strategy"
    optimizer = gsplat / "optimizers"
    strategy.mkdir(parents=True)
    optimizer.mkdir()
    (gsplat / "__init__.py").write_text("from .optimizers import SelectiveAdam\n")
    (strategy / "default.py").write_text("class DefaultStrategy:\n    absgrad: bool\n")
    (strategy / "mcmc.py").write_text("class MCMCStrategy:\n    pass\n")
    (optimizer / "selective_adam.py").write_text("class SelectiveAdam:\n    pass\n")
    (gsplat / "rendering.py").write_text(
        "def rasterization(*, packed: bool = False, sparse_grad: bool = False, "
        "absgrad: bool = False):\n    pass\n"
    )
    nerfstudio = site / "nerfstudio" / "models"
    nerfstudio.mkdir(parents=True)
    (nerfstudio / "splatfacto.py").write_text(
        "\n".join(
            [
                "from gsplat.strategy import DefaultStrategy",
                "use_absgrad: bool = True",
                "packed=False",
                "sparse_grad=False",
            ]
        )
    )
    _make_distribution(site, "gsplat", "1.5.3")
    _make_distribution(site, "nerfstudio", "1.1.5")

    repo = tmp_path / "repo"
    backend = repo / "backend"
    backend.mkdir(parents=True)
    (backend / "splat_route.py").write_text(
        'SFM_ESCALATION = ["colmap", "glomap"]\n'
        'command = "colmap global_mapper"\n'
        'gate = "splatlab-compute-gate.sh"\n'
    )

    result = capabilities.probe(
        repo,
        prefix,
        tmp_path / "missing-colmap",
    )

    assert result["features"]["mcmc_strategy_installed"] is True
    assert result["features"]["selective_adam_installed"] is True
    assert result["current_integration"]["nerfstudio_uses_mcmc_strategy"] is False
    assert result["current_integration"]["nerfstudio_absgrad_default_true"] is True
    assert result["current_integration"]["splatlab_global_mapper_configured"] is True
    statuses = {
        item["capability"]: item["status"] for item in result["recommendations"]
    }
    assert statuses["MCMC-densification"] == "library-present-adapter-required"
    assert statuses["absolute-gradient-densification"] == "already-enabled"


def test_distribution_metadata_is_read_without_importing_package(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    _make_distribution(site, "gsplat", "9.9.9")

    result = capabilities._distribution(site, "gsplat")

    assert result["installed"] is True
    assert result["version"] == "9.9.9"
    assert len(result["metadata_sha256"]) == 64
