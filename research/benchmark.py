#!/usr/bin/env python3
"""License-aware research catalog and benchmark receipt utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlparse


CATALOG_SCHEMA = "dev.splatlab.research-catalog/v1"
RECEIPT_SCHEMA = "dev.splatlab.research-benchmark/v1"
EXPORT_SCHEMA = "dev.splatlab.exports/v1"
EXPORT_REPORT_SCHEMA = "dev.splatlab.export-benchmark/v1"
ADAPTER_SCHEMA = "dev.splatlab.research-adapters/v1"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
DECISIONS = {"implemented", "evaluate", "watch", "deprioritize", "reject"}
INTEGRATION_MODES = {
    "production-permissive",
    "isolated-research",
    "paper-inspired-independent",
    "watch-only",
}
PRIORITIES = {"P0", "P1", "P2", "P3"}
CODE_STATUSES = {"available", "no-code", "placeholder"}
ROOT_ENV_RE = re.compile(r"^SPLATLAB_RESEARCH_[A-Z0-9_]+_ROOT$")
MAX_JSON_BYTES = 16 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[1]


class ResearchValidationError(ValueError):
    """A catalog, receipt, or export report violates its declared contract."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ResearchValidationError(f"JSON file is missing or unsafe: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ResearchValidationError(
            f"JSON file exceeds {MAX_JSON_BYTES} bytes: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchValidationError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ResearchValidationError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _https_url(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ResearchValidationError(f"{field} must be an HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ResearchValidationError(f"{field} must be an HTTPS URL")


def _nonempty_strings(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise ResearchValidationError(f"{field} must contain unique non-empty strings")
    return value


def validate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ResearchValidationError("Unsupported research catalog schema")
    policy = catalog.get("policy")
    if not isinstance(policy, dict):
        raise ResearchValidationError("Research catalog policy is missing")
    permissive = set(
        _nonempty_strings(policy.get("permissive_spdx"), "permissive_spdx")
    )
    if policy.get("restricted_code_location") != "outside-the-splatlab-repository":
        raise ResearchValidationError("Restricted code location policy is invalid")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ResearchValidationError("Research catalog has no candidates")

    ids: set[str] = set()
    decisions: dict[str, int] = {}
    modes: dict[str, int] = {}
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ResearchValidationError(f"Candidate {index} is not an object")
        candidate_id = item.get("id")
        if not isinstance(candidate_id, str) or not SAFE_ID_RE.fullmatch(candidate_id):
            raise ResearchValidationError(f"Candidate {index} has an unsafe id")
        if candidate_id in ids:
            raise ResearchValidationError(f"Duplicate candidate id: {candidate_id}")
        ids.add(candidate_id)
        if not isinstance(item.get("title"), str) or not item["title"]:
            raise ResearchValidationError(f"{candidate_id}: title is missing")
        _https_url(item.get("source_url"), f"{candidate_id}.source_url")
        _https_url(
            item.get("repository_url"),
            f"{candidate_id}.repository_url",
            nullable=True,
        )

        decision = item.get("decision")
        mode = item.get("integration_mode")
        priority = item.get("priority")
        if decision not in DECISIONS:
            raise ResearchValidationError(f"{candidate_id}: invalid decision")
        if mode not in INTEGRATION_MODES:
            raise ResearchValidationError(f"{candidate_id}: invalid integration mode")
        if priority not in PRIORITIES:
            raise ResearchValidationError(f"{candidate_id}: invalid priority")
        decisions[decision] = decisions.get(decision, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1

        license_record = item.get("license")
        if not isinstance(license_record, dict):
            raise ResearchValidationError(f"{candidate_id}: license record is missing")
        spdx = license_record.get("spdx")
        commercial = license_record.get("commercial_use")
        verified = license_record.get("verified")
        if not isinstance(spdx, str) or not spdx:
            raise ResearchValidationError(
                f"{candidate_id}: license identifier is missing"
            )
        if (
            commercial is not True
            and commercial is not False
            and commercial is not None
        ) or not isinstance(verified, bool):
            raise ResearchValidationError(f"{candidate_id}: invalid license assessment")

        code_status = item.get("code_status")
        if code_status not in CODE_STATUSES:
            raise ResearchValidationError(f"{candidate_id}: invalid code status")
        repository = item.get("repository_url")
        if code_status == "available" and repository is None:
            raise ResearchValidationError(
                f"{candidate_id}: available code has no repository"
            )
        if mode == "production-permissive":
            if (
                code_status != "available"
                or commercial is not True
                or verified is not True
                or spdx not in permissive
            ):
                raise ResearchValidationError(
                    f"{candidate_id}: production integration lacks a verified permissive license"
                )
        if commercial is False and mode != "isolated-research":
            raise ResearchValidationError(
                f"{candidate_id}: restricted code must remain isolated research"
            )
        if mode == "watch-only" and decision not in {"watch", "deprioritize", "reject"}:
            raise ResearchValidationError(
                f"{candidate_id}: watch-only source cannot be marked for implementation"
            )

        _nonempty_strings(item.get("techniques"), f"{candidate_id}.techniques")
        risks = item.get("risks")
        if not isinstance(risks, list) or not all(
            isinstance(risk, str) and risk for risk in risks
        ):
            raise ResearchValidationError(
                f"{candidate_id}: risks must be a string list"
            )
        gates = item.get("gates")
        if not isinstance(gates, list) or not all(
            isinstance(gate, str) and gate for gate in gates
        ):
            raise ResearchValidationError(
                f"{candidate_id}: gates must be a string list"
            )
        if decision == "evaluate" and not gates:
            raise ResearchValidationError(f"{candidate_id}: evaluations require gates")
        incorporation = item.get("incorporation")
        if not isinstance(incorporation, dict):
            raise ResearchValidationError(
                f"{candidate_id}: incorporation record is missing"
            )
        for name in ("implemented", "next"):
            values = incorporation.get(name)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ResearchValidationError(
                    f"{candidate_id}: incorporation.{name} must be a string list"
                )

    return {
        "status": "valid",
        "schema": CATALOG_SCHEMA,
        "candidates": len(candidates),
        "decisions": dict(sorted(decisions.items())),
        "integration_modes": dict(sorted(modes.items())),
        "p0": sorted(item["id"] for item in candidates if item.get("priority") == "P0"),
    }


def _catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_catalog(catalog)
    return {item["id"]: item for item in catalog["candidates"]}


def validate_adapter_registry(
    registry: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    if registry.get("schema") != ADAPTER_SCHEMA:
        raise ResearchValidationError("Unsupported research adapter schema")
    candidates = _catalog_index(catalog)
    adapters = registry.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ResearchValidationError("Research adapter registry is empty")
    seen: set[str] = set()
    restricted = 0
    gpu = 0
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            raise ResearchValidationError(f"Adapter {index} is not an object")
        adapter_id = adapter.get("id")
        if not isinstance(adapter_id, str) or not SAFE_ID_RE.fullmatch(adapter_id):
            raise ResearchValidationError(f"Adapter {index} has an unsafe id")
        if adapter_id in seen:
            raise ResearchValidationError(f"Duplicate research adapter: {adapter_id}")
        seen.add(adapter_id)
        source_id = adapter.get("source_id")
        if source_id not in candidates:
            raise ResearchValidationError(f"{adapter_id}: unknown catalog source")
        source = candidates[source_id]
        if (
            adapter.get("enabled") is not False
            or adapter.get("execution") != "plan-only"
        ):
            raise ResearchValidationError(
                f"{adapter_id}: research adapters must remain disabled and plan-only"
            )
        if adapter.get("checkout_policy") != "outside-splatlab-repository":
            raise ResearchValidationError(
                f"{adapter_id}: checkout must remain outside SplatLab"
            )
        root_env = adapter.get("root_env_var")
        if not isinstance(root_env, str) or not ROOT_ENV_RE.fullmatch(root_env):
            raise ResearchValidationError(
                f"{adapter_id}: invalid root environment variable"
            )
        license_scope = adapter.get("license_scope")
        expected_restricted = source["license"]["commercial_use"] is False
        if expected_restricted and license_scope != "restricted-research":
            raise ResearchValidationError(
                f"{adapter_id}: restricted source needs restricted-research scope"
            )
        if license_scope not in {"restricted-research", "permissive-evaluation"}:
            raise ResearchValidationError(f"{adapter_id}: invalid license scope")
        if license_scope == "restricted-research":
            restricted += 1

        resources = adapter.get("resources")
        if not isinstance(resources, dict):
            raise ResearchValidationError(f"{adapter_id}: resource policy is missing")
        uses_gpu = resources.get("gpu")
        if not isinstance(uses_gpu, bool):
            raise ResearchValidationError(f"{adapter_id}: gpu policy must be boolean")
        if uses_gpu:
            gpu += 1
            if resources.get("compute_gate_required") is not True:
                raise ResearchValidationError(
                    f"{adapter_id}: GPU adapters require the SplatLab compute gate"
                )
            _number(
                resources.get("minimum_vram_mb"),
                f"{adapter_id}.minimum_vram_mb",
                minimum=1,
                integer=True,
            )
        elif resources.get("minimum_vram_mb") not in {0, None}:
            raise ResearchValidationError(
                f"{adapter_id}: CPU adapter cannot reserve GPU memory"
            )
        if resources.get("network_during_run") is not False:
            raise ResearchValidationError(
                f"{adapter_id}: benchmark runs must be network-independent"
            )

        required_paths = adapter.get("required_paths")
        if not isinstance(required_paths, list) or not required_paths:
            raise ResearchValidationError(f"{adapter_id}: required_paths is empty")
        for relative in required_paths:
            _safe_artifact_path(relative)
        _nonempty_strings(adapter.get("accepts"), f"{adapter_id}.accepts")
        _nonempty_strings(adapter.get("produces"), f"{adapter_id}.produces")
        _nonempty_strings(adapter.get("manual_gates"), f"{adapter_id}.manual_gates")
        if "command" in adapter or "commands" in adapter:
            raise ResearchValidationError(
                f"{adapter_id}: adapter registry must not embed executable commands"
            )

    return {
        "status": "valid",
        "schema": ADAPTER_SCHEMA,
        "adapters": len(adapters),
        "restricted": restricted,
        "gpu": gpu,
        "ids": sorted(seen),
    }


def probe_adapter_registry(
    registry: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    summary = validate_adapter_registry(registry, catalog)
    probes = []
    for adapter in registry["adapters"]:
        root_value = os.environ.get(adapter["root_env_var"])
        root = Path(root_value).expanduser().resolve() if root_value else None
        inside_repository = False
        if root is not None:
            try:
                root.relative_to(REPO_ROOT)
                inside_repository = True
            except ValueError:
                pass
        present = []
        missing = []
        for relative in adapter["required_paths"]:
            exists = bool(root and root.is_dir() and (root / relative).exists())
            (present if exists else missing).append(relative)
        probes.append(
            {
                "id": adapter["id"],
                "source_id": adapter["source_id"],
                "configured": root is not None,
                "ready": root is not None and not missing and not inside_repository,
                "root_env_var": adapter["root_env_var"],
                "root": str(root) if root else None,
                "policy_violation": (
                    "checkout-is-inside-splatlab-repository"
                    if inside_repository
                    else None
                ),
                "present": present,
                "missing": missing,
                "execution": "plan-only",
            }
        )
    return {
        **summary,
        "status": "probed",
        "probes": probes,
    }


def _safe_artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ResearchValidationError("Artifact path must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not path.parts
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResearchValidationError(f"Unsafe artifact path: {value}")
    return path.as_posix()


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    integer: bool = False,
    nullable: bool = False,
) -> float | int | None:
    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ResearchValidationError(f"{field} must be a finite number")
    if integer and not isinstance(value, int):
        raise ResearchValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ResearchValidationError(f"{field} must be at least {minimum}")
    return value


def _timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ResearchValidationError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchValidationError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ResearchValidationError(f"{field} must include a timezone")


def validate_receipt(
    receipt: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ResearchValidationError("Unsupported benchmark receipt schema")
    experiment_id = receipt.get("experiment_id")
    if not isinstance(experiment_id, str) or not SAFE_ID_RE.fullmatch(experiment_id):
        raise ResearchValidationError("Benchmark experiment_id is unsafe")
    _timestamp(receipt.get("created_at"), "created_at")

    candidates = _catalog_index(catalog)
    adapter_id = receipt.get("adapter_id")
    if adapter_id not in candidates:
        raise ResearchValidationError(f"Unknown benchmark adapter: {adapter_id}")
    candidate = candidates[adapter_id]
    scope = receipt.get("scope")
    if scope not in {"production-evaluation", "isolated-research"}:
        raise ResearchValidationError("Invalid benchmark scope")
    if (
        candidate["integration_mode"] == "isolated-research"
        and scope != "isolated-research"
    ):
        raise ResearchValidationError(
            "Restricted adapter receipt must be research-only"
        )

    dataset = receipt.get("dataset")
    if not isinstance(dataset, dict):
        raise ResearchValidationError("Benchmark dataset record is missing")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not SAFE_ID_RE.fullmatch(dataset_id):
        raise ResearchValidationError("Benchmark dataset id is unsafe")
    if not SHA256_RE.fullmatch(str(dataset.get("input_sha256", ""))):
        raise ResearchValidationError("Benchmark dataset input SHA-256 is invalid")
    if not isinstance(dataset.get("capture_type"), str) or not dataset["capture_type"]:
        raise ResearchValidationError("Benchmark capture_type is missing")

    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        raise ResearchValidationError("Benchmark implementation record is missing")
    if implementation.get("kind") not in {"upstream", "independent"}:
        raise ResearchValidationError("Invalid benchmark implementation kind")
    _https_url(implementation.get("repository"), "implementation.repository")
    if not GIT_REVISION_RE.fullmatch(str(implementation.get("revision", ""))):
        raise ResearchValidationError("Implementation revision must be a full Git SHA")
    if not isinstance(implementation.get("dirty"), bool):
        raise ResearchValidationError("Implementation dirty flag must be boolean")
    if implementation["kind"] == "upstream":
        expected_repository = candidate.get("repository_url")
        if expected_repository and implementation["repository"].rstrip(
            "/"
        ) != expected_repository.rstrip("/"):
            raise ResearchValidationError("Receipt repository does not match catalog")

    environment = receipt.get("environment")
    if not isinstance(environment, dict):
        raise ResearchValidationError("Benchmark environment is missing")
    for field in ("os", "python", "gpu"):
        value = environment.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ResearchValidationError(
                f"environment.{field} must be a string or null"
            )
    _number(environment.get("vram_mb"), "environment.vram_mb", minimum=0, nullable=True)
    if not SHA256_RE.fullmatch(str(environment.get("dependencies_sha256", ""))):
        raise ResearchValidationError("Dependency lock SHA-256 is invalid")

    configuration = receipt.get("configuration")
    if not isinstance(configuration, dict) or not configuration:
        raise ResearchValidationError("Benchmark configuration is empty")

    outcome = receipt.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") not in {
        "pass",
        "fail",
        "incomplete",
    }:
        raise ResearchValidationError("Benchmark outcome is invalid")
    if not isinstance(outcome.get("notes"), str):
        raise ResearchValidationError("Benchmark outcome notes must be a string")
    if outcome["status"] != "pass" and not outcome["notes"].strip():
        raise ResearchValidationError("Failed or incomplete runs require notes")

    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        raise ResearchValidationError("Benchmark metrics are missing")
    _number(metrics.get("wall_seconds"), "metrics.wall_seconds", minimum=0)
    _number(metrics.get("peak_ram_mb"), "metrics.peak_ram_mb", minimum=0)
    _number(
        metrics.get("peak_vram_mb"),
        "metrics.peak_vram_mb",
        minimum=0,
        nullable=True,
    )
    output = metrics.get("output")
    if not isinstance(output, dict):
        raise ResearchValidationError("Benchmark output metrics are missing")
    _number(
        output.get("gaussian_count"),
        "metrics.output.gaussian_count",
        minimum=0,
        integer=True,
        nullable=True,
    )
    _number(
        output.get("bytes"),
        "metrics.output.bytes",
        minimum=0,
        integer=True,
        nullable=True,
    )
    quality = metrics.get("quality")
    if not isinstance(quality, dict):
        raise ResearchValidationError("Benchmark quality metrics must be an object")
    if outcome["status"] == "pass" and not quality:
        raise ResearchValidationError(
            "Passing benchmarks need at least one quality metric"
        )
    for name, value in quality.items():
        if not isinstance(name, str) or not name:
            raise ResearchValidationError("Quality metric names must be strings")
        _number(value, f"metrics.quality.{name}")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ResearchValidationError("Benchmark artifacts must be a list")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ResearchValidationError("Benchmark artifact must be an object")
        relative = _safe_artifact_path(artifact.get("path"))
        key = relative.casefold()
        if key in seen:
            raise ResearchValidationError(f"Duplicate benchmark artifact: {relative}")
        seen.add(key)
        if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
            raise ResearchValidationError(f"Invalid artifact SHA-256: {relative}")
        _number(artifact.get("bytes"), f"{relative}.bytes", minimum=0, integer=True)
        if not isinstance(artifact.get("role"), str) or not artifact["role"]:
            raise ResearchValidationError(f"Artifact role is missing: {relative}")

    return {
        "status": "valid",
        "schema": RECEIPT_SCHEMA,
        "experiment_id": experiment_id,
        "adapter_id": adapter_id,
        "dataset_id": dataset_id,
        "scope": scope,
        "quality_metrics": sorted(quality),
        "artifacts": len(artifacts),
        "outcome": outcome["status"],
    }


def _metric(receipt: dict[str, Any], dotted_path: str) -> float:
    current: Any = receipt
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ResearchValidationError(f"Metric path is missing: {dotted_path}")
        current = current[part]
    value = _number(current, dotted_path)
    assert isinstance(value, int | float)
    return float(value)


def compare_receipts(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    catalog: dict[str, Any],
    metric_path: str,
    direction: str,
    tolerance_percent: float,
) -> dict[str, Any]:
    validate_receipt(baseline, catalog)
    validate_receipt(candidate, catalog)
    if baseline["dataset"]["id"] != candidate["dataset"]["id"]:
        raise ResearchValidationError("Receipts use different datasets")
    if baseline["dataset"]["input_sha256"] != candidate["dataset"]["input_sha256"]:
        raise ResearchValidationError("Receipts use different dataset revisions")
    if direction not in {"higher", "lower"}:
        raise ResearchValidationError("Direction must be higher or lower")
    _number(tolerance_percent, "tolerance_percent", minimum=0)

    baseline_value = _metric(baseline, metric_path)
    candidate_value = _metric(candidate, metric_path)
    fraction = tolerance_percent / 100.0
    margin = abs(baseline_value) * fraction
    threshold = (
        baseline_value - margin if direction == "higher" else baseline_value + margin
    )
    passed = (
        candidate_value >= threshold
        if direction == "higher"
        else candidate_value <= threshold
    )
    delta = candidate_value - baseline_value
    delta_percent = None if baseline_value == 0 else delta / abs(baseline_value) * 100
    return {
        "status": "pass" if passed else "regression",
        "metric": metric_path,
        "direction": direction,
        "tolerance_percent": tolerance_percent,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "threshold": threshold,
        "delta": delta,
        "delta_percent": delta_percent,
        "baseline_experiment": baseline["experiment_id"],
        "candidate_experiment": candidate["experiment_id"],
    }


def export_report(
    manifest_path: Path, *, verify_files: bool = False, job_root: Path | None = None
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ResearchValidationError("Unsupported SplatLab export manifest schema")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id):
        raise ResearchValidationError("Export manifest job_id is unsafe")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ResearchValidationError("Export manifest source is missing")
    source_bytes = _number(source.get("bytes"), "source.bytes", minimum=1, integer=True)
    gaussian_count = _number(
        source.get("gaussian_count"),
        "source.gaussian_count",
        minimum=0,
        integer=True,
    )
    assert isinstance(source_bytes, int)
    assert isinstance(gaussian_count, int)
    source_sha = source.get("sha256")
    if not isinstance(source_sha, str) or not SHA256_RE.fullmatch(source_sha):
        raise ResearchValidationError("Canonical source SHA-256 is invalid")

    export_root = manifest_path.parent
    if job_root is None and export_root.name == "_exports":
        job_root = export_root.parent
    if job_root is not None:
        job_root = job_root.expanduser().resolve()
    checked_files = 0

    def records_report(records: Any) -> tuple[int, int]:
        nonlocal checked_files
        if not isinstance(records, list) or not records:
            raise ResearchValidationError("Ready export has no file records")
        total = 0
        for record in records:
            if not isinstance(record, dict):
                raise ResearchValidationError("Export file record is invalid")
            relative = _safe_artifact_path(record.get("path"))
            size = _number(
                record.get("bytes"), f"{relative}.bytes", minimum=0, integer=True
            )
            assert isinstance(size, int)
            expected_sha = record.get("sha256")
            if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(
                expected_sha
            ):
                raise ResearchValidationError(f"Invalid export SHA-256: {relative}")
            total += size
            if verify_files:
                path = (export_root / relative).resolve()
                try:
                    path.relative_to(export_root)
                except ValueError as exc:
                    raise ResearchValidationError(
                        f"Export file escapes root: {relative}"
                    ) from exc
                if not path.is_file() or path.is_symlink():
                    raise ResearchValidationError(f"Export file is missing: {relative}")
                if path.stat().st_size != size or _sha256(path) != expected_sha.lower():
                    raise ResearchValidationError(
                        f"Export file identity mismatch: {relative}"
                    )
                checked_files += 1
        return len(records), total

    artifacts = []
    derived_bytes = 0
    for name, artifact in sorted(manifest.get("artifacts", {}).items()):
        if not isinstance(artifact, dict):
            continue
        record = {"name": name, "status": artifact.get("status")}
        if artifact.get("status") == "ready":
            file_count, total = records_report(artifact.get("files"))
            record.update(
                {
                    "file_count": file_count,
                    "bytes": total,
                    "source_ratio": total / source_bytes,
                    "bytes_per_gaussian": (
                        total / gaussian_count if gaussian_count else None
                    ),
                    "parameters": artifact.get("parameters"),
                }
            )
            derived_bytes += total
        elif artifact.get("status") == "skipped":
            record["reason"] = artifact.get("reason")
        artifacts.append(record)

    collision = manifest.get("collision")
    collision_report = None
    if isinstance(collision, dict) and collision.get("status") == "ready":
        file_count, total = records_report(collision.get("files"))
        collision_report = {
            "status": "ready",
            "mode": collision.get("mode"),
            "file_count": file_count,
            "bytes": total,
            "source_ratio": total / source_bytes,
        }
        derived_bytes += total

    if verify_files:
        if job_root is None:
            raise ResearchValidationError(
                "Cannot locate source file; provide --job-root"
            )
        source_relative = _safe_artifact_path(source.get("path"))
        source_path = (job_root / source_relative).resolve()
        try:
            source_path.relative_to(job_root)
        except ValueError as exc:
            raise ResearchValidationError("Source file escapes job root") from exc
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or source_path.stat().st_size != source_bytes
            or _sha256(source_path) != source_sha.lower()
        ):
            raise ResearchValidationError("Canonical source file identity mismatch")
        checked_files += 1

    return {
        "schema": EXPORT_REPORT_SCHEMA,
        "job_id": job_id,
        "manifest": str(manifest_path),
        "source": {
            "bytes": source_bytes,
            "gaussian_count": gaussian_count,
            "bytes_per_gaussian": (
                source_bytes / gaussian_count if gaussian_count else None
            ),
            "sha256": source_sha,
        },
        "artifacts": artifacts,
        "collision": collision_report,
        "ready_derived_bytes": derived_bytes,
        "checked_files": checked_files,
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


def _emit(value: dict[str, Any], output: Path | None = None) -> None:
    if output:
        _atomic_json(output, value)
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _default_catalog() -> Path:
    return Path(__file__).with_name("sources.json")


def _default_adapters() -> Path:
    return Path(__file__).with_name("adapters.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SplatLab research sources and benchmark receipts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("validate-catalog")
    catalog.add_argument("catalog", nargs="?", type=Path, default=_default_catalog())

    receipt = commands.add_parser("validate-receipt")
    receipt.add_argument("receipt", nargs="+", type=Path)
    receipt.add_argument("--catalog", type=Path, default=_default_catalog())

    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--catalog", type=Path, default=_default_catalog())
    compare.add_argument("--metric", required=True)
    compare.add_argument("--direction", choices=("higher", "lower"), required=True)
    compare.add_argument("--tolerance-percent", type=float, default=0)

    exports = commands.add_parser("exports")
    exports.add_argument("manifest", type=Path)
    exports.add_argument("--verify-files", action="store_true")
    exports.add_argument("--job-root", type=Path)
    exports.add_argument("--output", type=Path)

    adapters = commands.add_parser("validate-adapters")
    adapters.add_argument("registry", nargs="?", type=Path, default=_default_adapters())
    adapters.add_argument("--catalog", type=Path, default=_default_catalog())

    probe = commands.add_parser("probe-adapters")
    probe.add_argument("registry", nargs="?", type=Path, default=_default_adapters())
    probe.add_argument("--catalog", type=Path, default=_default_catalog())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-catalog":
            catalog = _load_json(args.catalog)
            result = validate_catalog(catalog)
            result["path"] = str(args.catalog.expanduser().resolve())
            result["sha256"] = _sha256(args.catalog.expanduser().resolve())
            _emit(result)
        elif args.command == "validate-receipt":
            catalog = _load_json(args.catalog)
            results = [
                validate_receipt(_load_json(path), catalog) for path in args.receipt
            ]
            _emit(
                {
                    "status": "valid",
                    "schema": RECEIPT_SCHEMA,
                    "receipts": results,
                }
            )
        elif args.command == "compare":
            catalog = _load_json(args.catalog)
            result = compare_receipts(
                _load_json(args.baseline),
                _load_json(args.candidate),
                catalog,
                args.metric,
                args.direction,
                args.tolerance_percent,
            )
            _emit(result)
            if result["status"] != "pass":
                return 3
        elif args.command == "exports":
            _emit(
                export_report(
                    args.manifest,
                    verify_files=args.verify_files,
                    job_root=args.job_root,
                ),
                args.output,
            )
        elif args.command == "validate-adapters":
            _emit(
                validate_adapter_registry(
                    _load_json(args.registry), _load_json(args.catalog)
                )
            )
        elif args.command == "probe-adapters":
            _emit(
                probe_adapter_registry(
                    _load_json(args.registry), _load_json(args.catalog)
                )
            )
        else:  # pragma: no cover - argparse requires a command.
            raise ResearchValidationError(f"Unknown command: {args.command}")
    except ResearchValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
