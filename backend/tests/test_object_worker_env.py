"""Tests for the model-worker subprocess contract in mesh/object_generate.py.

Two guarantees are pinned here:

1. **No secret reaches a model runtime.** This process is launched from a shell
   that has had the Vaultwarden secrets injected into it. `run_worker` starts
   three third-party model environments (SAM 3, SA-3DAO, the nerfstudio probe);
   inheriting `os.environ` wholesale handed every one of them the API keys, the
   database URL and the session token. `dcc/blender_workflow` already
   allowlisted its Blender subprocesses — this is the same posture for the
   generative lane.

2. **Success is structural, not narrative.** A worker "succeeded" if it exited 0
   AND the artifact it was asked for is on disk. Grepping a 4 KB log tail for a
   done-token is the fallback, not the proof.

Real subprocesses are used (a throwaway python script), so the env and the exit
handling are exercised end to end rather than mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

import object_generate as og  # noqa: E402


SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-secret",
    "GEMINI_API_KEY": "gem-secret",
    "PORTAL_TOKEN": "portal-secret",
    "DATABASE_URL": "postgres://user:pw@host/db",
    "REDIS_PASSWORD": "redis-secret",
    "LITELLM_MASTER_KEY": "sk-master",
    "BW_SESSION": "bw-secret",
    "AWS_SECRET_ACCESS_KEY": "aws-secret",
}


# ---------------------------------------------------------------------------
# worker_env — the allowlist
# ---------------------------------------------------------------------------

def test_worker_env_drops_every_vault_secret(monkeypatch):
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)

    env = og.worker_env()

    for key in SECRETS:
        assert key not in env, f"{key} leaked into a model subprocess"
    assert not any(v in env.values() for v in SECRETS.values())


def test_worker_env_keeps_the_compute_variables_the_models_need(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64")
    monkeypatch.setenv("HF_HOME", "/home/rtoony/.cache/huggingface")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("TORCH_HOME", "/home/rtoony/.cache/torch")

    env = og.worker_env()

    assert env["CUDA_HOME"] == "/opt/cuda"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"
    assert env["HF_HOME"] == "/home/rtoony/.cache/huggingface"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["TORCH_HOME"] == "/home/rtoony/.cache/torch"


def test_worker_env_always_sets_the_isolation_defaults(monkeypatch):
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.delenv("PATH", raising=False)

    env = og.worker_env()

    assert env["PYTHONNOUSERSITE"] == "1", "user site-packages must never leak in"
    assert env["PATH"]
    assert env["HOME"]


def test_worker_env_extra_is_applied_and_stringified(monkeypatch):
    env = og.worker_env({"SAM3D_ROOT": Path("/home/rtoony/tools/sam-3d-objects")})
    assert env["SAM3D_ROOT"] == "/home/rtoony/tools/sam-3d-objects"


def test_worker_env_extra_can_override_an_inherited_value(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/wrong/env")
    env = og.worker_env({"CONDA_PREFIX": "/right/env"})
    assert env["CONDA_PREFIX"] == "/right/env"


def test_secret_prefix_rules_cannot_match_a_nexus_secret():
    """The prefix allowlist is only safe while no secret starts with one."""
    for key in SECRETS:
        assert not key.startswith(og.WORKER_ENV_ALLOW_PREFIXES)
        assert key not in og.WORKER_ENV_ALLOW


# ---------------------------------------------------------------------------
# run_worker — structural success detection
# ---------------------------------------------------------------------------

def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "worker_fake.py"
    path.write_text("import json, os, sys\n" + body)
    return path


def _run(tmp_path: Path, script: Path, **kwargs):
    return og.run_worker(Path(sys.executable), script, [], tmp_path,
                         tmp_path / "worker.log", timeout=60, **kwargs)


def test_run_worker_raises_on_nonzero_exit(tmp_path):
    script = _script(tmp_path, "print('boom'); sys.exit(3)")
    with pytest.raises(og.Fatal, match="exited 3"):
        _run(tmp_path, script)


def test_run_worker_accepts_a_declared_artifact(tmp_path):
    out = tmp_path / "result.json"
    script = _script(tmp_path, f"open({str(out)!r},'w').write(json.dumps({{'ok':1}}))")
    _run(tmp_path, script, produces=out)          # must not raise


def test_run_worker_rejects_a_missing_artifact(tmp_path):
    script = _script(tmp_path, "print('WORKER_DONE')")
    with pytest.raises(og.Fatal, match="never wrote"):
        _run(tmp_path, script, produces=tmp_path / "result.json")


def test_run_worker_rejects_an_empty_artifact(tmp_path):
    out = tmp_path / "masks.npz"
    script = _script(tmp_path, f"open({str(out)!r},'w').close()")
    with pytest.raises(og.Fatal, match="is empty"):
        _run(tmp_path, script, produces=out)


def test_run_worker_rejects_truncated_json(tmp_path):
    """A worker killed mid-write leaves valid-looking output on disk."""
    out = tmp_path / "result.json"
    script = _script(tmp_path, f"open({str(out)!r},'w').write('[{{\"slug\": ')")
    with pytest.raises(og.Fatal, match="not valid JSON"):
        _run(tmp_path, script, produces=out)


def test_run_worker_checks_every_artifact_in_a_list(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    script = _script(tmp_path, f"open({str(first)!r},'w').write('x')")
    with pytest.raises(og.Fatal, match="b.npz"):
        _run(tmp_path, script, produces=[first, second])


def test_declared_artifact_beats_a_missing_done_token(tmp_path):
    """The regression this replaces: a worker that produced correct output but
    whose done-token was pushed out of the 4 KB tail by late CUDA warnings."""
    out = tmp_path / "result.json"
    script = _script(tmp_path, (
        f"open({str(out)!r},'w').write(json.dumps({{'ok':1}}))\n"
        "print('WORKER_SAM3D_DONE')\n"
        "print('cuda warning ' * 900)\n"          # > 4 KB of noise after the token
    ))
    _run(tmp_path, script, produces=out, done_token="WORKER_SAM3D_DONE")

    tail = (tmp_path / "worker.log").read_text()[-4000:]
    assert "WORKER_SAM3D_DONE" not in tail, "test must actually push the token out"


def test_done_token_still_guards_workers_with_no_declared_artifact(tmp_path):
    script = _script(tmp_path, "print('finished quietly')")
    with pytest.raises(og.Fatal, match="never printed WORKER_DONE"):
        _run(tmp_path, script, done_token="WORKER_DONE")


def test_run_worker_passes_only_allowlisted_env_to_the_child(tmp_path, monkeypatch):
    """End to end: the child process itself must not see the secrets."""
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    out = tmp_path / "seen.json"
    script = _script(tmp_path, f"open({str(out)!r},'w').write(json.dumps(dict(os.environ)))")
    _run(tmp_path, script, produces=out)

    child = json.loads(out.read_text())
    for key in SECRETS:
        assert key not in child, f"{key} reached the model subprocess"
    assert child["CUDA_VISIBLE_DEVICES"] == "0"
    assert child["PYTHONNOUSERSITE"] == "1"
