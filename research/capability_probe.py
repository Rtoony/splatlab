#!/usr/bin/env python3
"""Static compatibility probe for SplatLab's installed reconstruction stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "dev.splatlab.research-capabilities/v1"
MAX_SOURCE_BYTES = 16 * 1024 * 1024


class CapabilityProbeError(ValueError):
    """The requested environment cannot be inspected safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise CapabilityProbeError(f"Source file is unexpectedly large: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CapabilityProbeError(f"Cannot read source file: {path}") from exc


def _site_packages(prefix: Path) -> Path:
    candidates = sorted(prefix.glob("lib/python*/site-packages"))
    if not candidates:
        raise CapabilityProbeError(f"No Python site-packages found under {prefix}")

    def version_key(path: Path) -> tuple[int, ...]:
        raw = path.parent.name.removeprefix("python")
        try:
            return tuple(int(part) for part in raw.split("."))
        except ValueError:
            return (0,)

    return max(candidates, key=version_key)


def _distribution(site_packages: Path, normalized_name: str) -> dict[str, Any]:
    candidates = sorted(
        site_packages.glob(f"{normalized_name.replace('-', '_')}-*.dist-info/METADATA")
    )
    if len(candidates) != 1:
        return {
            "installed": False,
            "version": None,
            "metadata": None,
        }
    metadata_path = candidates[0]
    metadata = _text(metadata_path)
    version = None
    for line in metadata.splitlines():
        if line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
            break
    return {
        "installed": True,
        "version": version,
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def _source_record(path: Path, checks: dict[str, bool]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
        "checks": checks,
    }


def _colmap_version(binary: Path) -> str | None:
    if not binary.is_file():
        return None
    env = {
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }
    try:
        result = subprocess.run(
            [str(binary), "-h"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
            env=env,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CapabilityProbeError(f"COLMAP help probe failed: {exc}") from exc
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line or None


def probe(
    repo_root: Path,
    splatops_prefix: Path,
    colmap4_binary: Path,
    *,
    probe_colmap_cli: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    splatops_prefix = splatops_prefix.expanduser().resolve()
    colmap4_binary = colmap4_binary.expanduser().resolve()
    site_packages = _site_packages(splatops_prefix)

    gsplat = _distribution(site_packages, "gsplat")
    nerfstudio = _distribution(site_packages, "nerfstudio")
    gsplat_root = site_packages / "gsplat"
    nerfstudio_model = site_packages / "nerfstudio" / "models" / "splatfacto.py"
    splatlab_route = repo_root / "backend" / "splat_route.py"
    if not splatlab_route.is_file():
        raise CapabilityProbeError(f"SplatLab backend not found: {repo_root}")

    gsplat_init = _text(gsplat_root / "__init__.py")
    gsplat_default = _text(gsplat_root / "strategy" / "default.py")
    gsplat_mcmc = _text(gsplat_root / "strategy" / "mcmc.py")
    gsplat_rendering = _text(gsplat_root / "rendering.py")
    nerfstudio_text = _text(nerfstudio_model)
    splatlab_text = _text(splatlab_route)

    features = {
        "mcmc_strategy_installed": "class MCMCStrategy" in gsplat_mcmc,
        "default_strategy_installed": "class DefaultStrategy" in gsplat_default,
        "selective_adam_installed": "SelectiveAdam" in gsplat_init
        and (gsplat_root / "optimizers" / "selective_adam.py").is_file(),
        "packed_rasterization_installed": "packed:" in gsplat_rendering
        or "packed =" in gsplat_rendering,
        "sparse_gradients_installed": "sparse_grad:" in gsplat_rendering,
        "absgrad_installed": "absgrad:" in gsplat_rendering,
    }
    integration = {
        "nerfstudio_uses_default_strategy": "from gsplat.strategy import DefaultStrategy"
        in nerfstudio_text,
        "nerfstudio_uses_mcmc_strategy": "MCMCStrategy" in nerfstudio_text,
        "nerfstudio_absgrad_default_true": bool(
            re.search(r"use_absgrad:\s*bool\s*=\s*True", nerfstudio_text)
        ),
        "nerfstudio_packed_hardcoded_false": "packed=False" in nerfstudio_text,
        "nerfstudio_sparse_grad_hardcoded_false": "sparse_grad=False"
        in nerfstudio_text,
        "splatlab_global_mapper_configured": bool(
            re.search(r"\bglobal_mapper\b", splatlab_text)
        ),
        "splatlab_solver_escalation_configured": "SFM_ESCALATION" in splatlab_text,
        "splatlab_compute_gate_referenced": "splatlab-compute-gate.sh" in splatlab_text,
    }

    recommendations = []
    if integration["splatlab_global_mapper_configured"]:
        recommendations.append(
            {
                "capability": "COLMAP-global-mapper",
                "status": "already-integrated",
                "action": "retain-current-glomap-rescue-and-rig-escalation",
            }
        )
    if integration["nerfstudio_absgrad_default_true"]:
        recommendations.append(
            {
                "capability": "absolute-gradient-densification",
                "status": "already-enabled",
                "action": "do-not-add-a-duplicate-profile",
            }
        )
    if (
        features["mcmc_strategy_installed"]
        and not integration["nerfstudio_uses_mcmc_strategy"]
    ):
        recommendations.append(
            {
                "capability": "MCMC-densification",
                "status": "library-present-adapter-required",
                "action": "evaluate-as-a-separate-Nerfstudio-method-before-UI-exposure",
            }
        )
    if (
        features["packed_rasterization_installed"]
        and features["sparse_gradients_installed"]
        and integration["nerfstudio_packed_hardcoded_false"]
        and integration["nerfstudio_sparse_grad_hardcoded_false"]
    ):
        recommendations.append(
            {
                "capability": "packed-sparse-training",
                "status": "library-present-model-change-required",
                "action": "benchmark-packed-and-sparse-gradients-together-in-an-isolated-model",
            }
        )
    if features["selective_adam_installed"]:
        recommendations.append(
            {
                "capability": "SelectiveAdam",
                "status": "library-present-integration-required",
                "action": "evaluate-only-with-a-compatible-packed-visibility-mask",
            }
        )

    colmap = {
        "path": str(colmap4_binary),
        "available": colmap4_binary.is_file(),
        "sha256": _sha256(colmap4_binary) if colmap4_binary.is_file() else None,
        "version": _colmap_version(colmap4_binary) if probe_colmap_cli else None,
        "cli_executed": probe_colmap_cli,
    }
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "static-source-inspection"
        + ("+colmap-help" if probe_colmap_cli else ""),
        "environment": {
            "splatops_prefix": str(splatops_prefix),
            "site_packages": str(site_packages),
            "gsplat": gsplat,
            "nerfstudio": nerfstudio,
            "colmap4": colmap,
        },
        "features": features,
        "current_integration": integration,
        "sources": {
            "gsplat_default": _source_record(
                gsplat_root / "strategy" / "default.py",
                {"class": features["default_strategy_installed"]},
            ),
            "gsplat_mcmc": _source_record(
                gsplat_root / "strategy" / "mcmc.py",
                {"class": features["mcmc_strategy_installed"]},
            ),
            "nerfstudio_splatfacto": _source_record(
                nerfstudio_model,
                {
                    "default_strategy": integration["nerfstudio_uses_default_strategy"],
                    "mcmc_strategy": integration["nerfstudio_uses_mcmc_strategy"],
                },
            ),
            "splatlab_route": _source_record(
                splatlab_route,
                {
                    "global_mapper": integration["splatlab_global_mapper_configured"],
                    "solver_escalation": integration[
                        "splatlab_solver_escalation_configured"
                    ],
                },
            ),
        },
        "recommendations": recommendations,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Inspect installed SplatLab capabilities without importing CUDA."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).parents[1],
    )
    parser.add_argument(
        "--splatops-prefix",
        type=Path,
        default=home / "miniconda3" / "envs" / "splatops",
    )
    parser.add_argument(
        "--colmap4-bin",
        type=Path,
        default=home / "miniconda3" / "envs" / "colmap4" / "bin" / "colmap",
    )
    parser.add_argument("--probe-colmap-cli", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = probe(
            args.repo_root,
            args.splatops_prefix,
            args.colmap4_bin,
            probe_colmap_cli=args.probe_colmap_cli,
        )
        if args.output:
            _atomic_json(args.output, result)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    except CapabilityProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
