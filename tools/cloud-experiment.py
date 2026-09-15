#!/usr/bin/env python3
"""Explicit overnight cloud experiments against one non-resettable $20 ledger."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from cloud_budget import CloudBudget
import cloud_experiments as cloud

RUN = ROOT / "data/spatial/cloud-overnight-2026-09-07"
DEADLINE = int(datetime(2026, 9, 7, 13, 40, 25, tzinfo=timezone.utc).timestamp())


def staged_file(name, suffix):
    if not cloud.re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}" + cloud.re.escape(suffix), name):
        raise ValueError("Use a staged basename, not an arbitrary path")
    path = RUN / "inputs" / name
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Input must be a regular, explicitly staged local file")
    return path


def retain_assets(output, response):
    texts, decoded = cloud.response_assets(response)
    images = []
    files = {output / "answer.txt": "\n\n".join(texts).encode()}
    for index, item in enumerate(decoded):
        path = output / f"candidate-{index:02d}.{item['extension']}"
        files[path] = item["content"]
        images.append({"path": str(path), "sha256": item["sha256"], "mime_type": item["mime_type"]})
    for path, content in files.items():
        if path.resolve() != path.absolute() or path.exists() and path.read_bytes() != content:
            raise ValueError("Retained result differs; do not overwrite evidence")
    for path, content in files.items():
        if not path.exists():
            path.write_bytes(content)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "status", "inventory", "generate", "collect"))
    parser.add_argument("--request-id")
    parser.add_argument("--model", choices=tuple(cloud.MODELS))
    parser.add_argument("--prompt")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()
    budget = CloudBudget(RUN / "spend.sqlite3")
    if args.action == "initialize":
        result = budget.initialize(20_000_000, DEADLINE)
    elif args.action == "status":
        result = budget.status()
    elif args.action == "inventory":
        result = cloud.inventory(cloud.vault_keys())
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "model-inventory.json").write_text(json.dumps(result, indent=2) + "\n")
        result = {"key_present": result["key_present"], "catalogs": {provider: {"status": info["status"], "model_count": len(info.get("models", []))} for provider, info in result["catalogs"].items()}, "report": str(RUN / "model-inventory.json")}
    elif args.action == "collect":
        if not args.request_id or not cloud.re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.request_id):
            parser.error("collect requires an existing request id; it never calls an API")
        record = next((item for item in budget.status()["requests"] if item["request_id"] == args.request_id), None)
        if not record or record["status"] != "returned":
            raise ValueError("No completed response is registered")
        output = RUN / "results" / args.request_id
        source = output / "response.json"
        if source.resolve() != source.absolute() or source.stat().st_size > cloud.MAX_RESPONSE_BYTES * 2:
            raise ValueError("Invalid retained response path or size")
        response = json.loads(source.read_bytes())
        digest = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
        if digest != record["result"]["response_sha256"]:
            raise ValueError("Retained response does not match the ledger")
        result = {"request_id": args.request_id, "response_sha256": digest, "images": retain_assets(output, response),
                  "api_calls": 0, "status": "needs-review", "render_vr_only": True, "not_geometric_evidence": True}
        target = output / "collection.json"
        encoded = json.dumps(result, indent=2) + "\n"
        if target.resolve() != target.absolute() or target.exists() and target.read_text() != encoded:
            raise ValueError("Existing collection evidence differs")
        if not target.exists():
            target.write_text(encoded)
    else:
        if not args.request_id or not args.model or not args.prompt:
            parser.error("generate requires --request-id, --model and --prompt staged-name.txt")
        budget.require_open()
        prompt_path = staged_file(args.prompt, ".txt")
        if prompt_path.stat().st_size > 32_768:
            raise ValueError("Prompt exceeds the upload limit")
        references = []
        for name in args.reference:
            path = staged_file(name, ".png")
            if path.stat().st_size > 4 * 1024 ** 2:
                raise ValueError("Reference exceeds the upload limit")
            references.append(path.read_bytes())
        output = RUN / "results" / args.request_id
        if not cloud.re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.request_id):
            raise ValueError("Invalid request id")
        output.mkdir(parents=True, exist_ok=False)
        prompt = prompt_path.read_text()
        (output / "prompt.txt").write_text(prompt)
        input_records = {"prompt": {"staged_name": args.prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()}, "references": []}
        for index, content in enumerate(references):
            name = f"reference-{index:02d}.png"
            (output / name).write_bytes(content)
            input_records["references"].append({"staged_name": args.reference[index], "retained_name": name, "sha256": hashlib.sha256(content).hexdigest()})
        (output / "inputs.json").write_text(json.dumps(input_records, indent=2) + "\n")
        result = cloud.generate(budget, cloud.vault_keys(), args.request_id, args.model,
                                prompt, args.max_output_tokens, references)
        response = result.pop("response")
        raw = json.dumps(response, indent=2).encode()
        (output / "response.json").write_bytes(raw)
        images = retain_assets(output, response)
        result.update({"output": str(output), "images": images, "status": "needs-review", "render_vr_only": True,
                       "not_geometric_evidence": True, "cloud_budget": budget.status()})
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.action == "generate":
        result["cloud_budget"] = {name: value for name, value in result["cloud_budget"].items() if name != "requests"}
    print(json.dumps(result, indent=2))


def timed_out(_number, _frame):
    raise cloud.ProviderFailure("wall-clock-timeout")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(240)
    try:
        main()
    except Exception as error:
        if isinstance(error, cloud.ProviderFailure):
            print(json.dumps({"error": error.category, "http_status": error.status}), file=sys.stderr)
        else:
            print(json.dumps({"error": type(error).__name__, "detail": "Experiment refused or failed; any existing cost reservation remains held"}), file=sys.stderr)
        raise SystemExit(1)
