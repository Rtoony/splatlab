"""Small, explicit provider experiments; no tools, retries, routing, or scene mutations."""

import base64
from decimal import Decimal, ROUND_CEILING
import hashlib
from io import BytesIO
import json
import re
import shlex
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image

from cloud_budget import BudgetRefused

PRICING_CHECKED_UTC = "2026-09-07"
PRICING_EXPIRES_UTC = 1788825600
PRICING_SOURCES = {
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
    "zai": "https://docs.z.ai/guides/overview/pricing",
    "kimi": "https://forum.moonshot.ai/t/kimi-k3-is-here-our-most-capable-model/480",
}
MODELS = {
    "gemini-3.8-flash": {"provider": "gemini", "input_limit": 1_048_576, "input_usd_per_million": "0.75", "output_usd_per_million": "3.75", "image_output": False},
    "gemini-3.1-flash-image": {"provider": "gemini", "input_limit": 131_072, "input_usd_per_million": "0.50", "output_usd_per_million": "60", "image_output": True},
    "gemini-3-pro-image": {"provider": "gemini", "input_limit": 131_072, "input_usd_per_million": "2", "output_usd_per_million": "120", "image_output": True},
    "glm-5.3": {"provider": "zai", "input_limit": 1_048_576, "input_usd_per_million": "1.4", "output_usd_per_million": "4.4", "image_output": False},
    "glm-5.3-flash": {"provider": "zai", "input_limit": 1_048_576, "input_usd_per_million": "0.15", "output_usd_per_million": "0.50", "image_output": False},
    "kimi-k3": {"provider": "kimi", "input_limit": 1_048_576, "input_usd_per_million": "3", "output_usd_per_million": "15", "image_output": False},
}
KEY_NAMES = ("GEMINI_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY", "OPENROUTER_API_KEY", "FIREWORKS_API_KEY", "ANTHROPIC_API_KEY")
PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "zai": "ZAI_API_KEY", "kimi": "KIMI_API_KEY"}
MAX_RESPONSE_BYTES = 24 * 1024 ** 2


class ProviderFailure(RuntimeError):
    def __init__(self, category, status=None):
        self.category, self.status = category, status
        super().__init__(f"Provider request failed: {category}" + (f" (HTTP {status})" if status else ""))


def vault_keys():
    try:
        result = subprocess.run(["/home/rtoony/bin/nexus-inject", "--all", "--quiet", "--cache", "300"],
                                capture_output=True, text=True, timeout=75, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ProviderFailure("vault-unavailable") from None
    keys, status = {}, None
    for line in result.stdout.splitlines():
        if not re.match(r"export (?:" + "|".join(KEY_NAMES) + r"|NEXUS_INJECT_STATUS)=", line):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            raise ProviderFailure("vault-export-format") from None
        if len(parts) != 2 or parts[0] != "export":
            raise ProviderFailure("vault-export-format")
        name, value = parts[1].split("=", 1)
        if name == "NEXUS_INJECT_STATUS":
            status = value
        elif value and re.fullmatch(r"[A-Za-z0-9_.:/+=-]+", value):
            keys[name] = value
    if result.returncode or status != "ok":
        raise ProviderFailure("vault-unavailable-unlock-with-nexus-unlock")
    return keys


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise ProviderFailure("redirect-refused", code)


def request_json(provider, key, action, payload=None):
    if provider == "gemini" and (action == "catalog" or re.fullmatch(r"models/[a-z0-9.-]+:(?:generateContent|countTokens)", action)):
        suffix = "models?pageSize=1000" if action == "catalog" else action
        url = "https://generativelanguage.googleapis.com/v1beta/" + suffix
        headers = {"x-goog-api-key": key}
    elif provider == "zai" and action in {"catalog", "chat/completions"}:
        url = "https://api.z.ai/api/paas/v4/" + ("models" if action == "catalog" else action)
        headers = {"Authorization": "Bearer " + key}
    elif provider == "kimi" and action in {"catalog", "chat/completions", "tokenizers/estimate-token-count"}:
        url = "https://api.moonshot.ai/v1/" + ("models" if action == "catalog" else action)
        headers = {"Authorization": "Bearer " + key}
    elif provider == "openrouter" and action == "catalog":
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": "Bearer " + key}
    else:
        raise ProviderFailure("endpoint-not-allowed")
    headers["Content-Type"] = "application/json"
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers=headers)
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=180) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderFailure("response-size-limit")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ProviderFailure("response-format")
        return parsed
    except HTTPError as error:
        raise ProviderFailure("http-error", error.code) from None
    except (URLError, OSError, ValueError):
        raise ProviderFailure("transport-or-response-error") from None


def inventory(keys):
    result = {"key_present": {name: name in keys for name in KEY_NAMES}, "catalogs": {},
              "note": "Catalog listing is not proof of paid quota or generation entitlement"}
    for provider, name in ((*PROVIDER_KEYS.items(), ("openrouter", "OPENROUTER_API_KEY"))):
        if name not in keys:
            continue
        try:
            catalog = request_json(provider, keys[name], "catalog")
            records = catalog.get("models", catalog.get("data", []))
            result["catalogs"][provider] = {"status": "listed", "truncated": bool(catalog.get("nextPageToken")), "models": [
                {field: entry[field] for field in ("id", "name", "inputTokenLimit", "outputTokenLimit", "supportedGenerationMethods", "pricing", "architecture") if field in entry}
                for entry in records if isinstance(entry, dict)
            ]}
        except ProviderFailure as error:
            result["catalogs"][provider] = {"status": error.category, "http_status": error.status}
    return result


def quote(model, output_tokens, input_tokens=None):
    if model not in MODELS or time.time() >= PRICING_EXPIRES_UTC:
        raise BudgetRefused("Model price is unverified or this short-lived rate card expired")
    if type(output_tokens) is not int or not 1 <= output_tokens <= 8192:
        raise BudgetRefused("Output token limit must be 1–8192")
    policy = MODELS[model]
    if input_tokens is None:
        input_tokens = policy["input_limit"]
    if type(input_tokens) is not int or not 0 < input_tokens <= policy["input_limit"]:
        raise BudgetRefused("Invalid or oversized input token count")
    cost = Decimal(input_tokens) * Decimal(policy["input_usd_per_million"]) + Decimal(output_tokens) * Decimal(policy["output_usd_per_million"])
    amount = int((cost * 2).to_integral_value(rounding=ROUND_CEILING)) + 20_000
    return amount, {**policy, "input_tokens_bound": input_tokens, "max_output_tokens": output_tokens,
                    "candidate_count": 1, "tools_enabled": False, "automatic_retries": 0,
                    "pricing_checked_utc": PRICING_CHECKED_UTC, "pricing_source": PRICING_SOURCES[policy["provider"]],
                    "reservation_policy": "2x quoted maximum plus $0.02; never released tonight"}


def image_part(raw):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 4 * 1024 ** 2:
        raise BudgetRefused("Reference must be a PNG no larger than 4 MiB")
    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.format != "PNG" or max(opened.size) > 2048 or min(opened.size) < 16:
                raise BudgetRefused("Only bounded, explicitly staged PNG reference crops are allowed")
            opened.verify()
    except (OSError, ValueError, Image.DecompressionBombError):
        raise BudgetRefused("Invalid PNG reference") from None
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(raw).decode()}}


def response_assets(response):
    texts, images = [], []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            inline = part.get("inlineData")
            if inline is None:
                continue
            mime = inline.get("mimeType")
            formats = {"image/png": ("PNG", "png"), "image/jpeg": ("JPEG", "jpg")}
            if mime not in formats or images:
                raise ProviderFailure("unexpected-image-output")
            try:
                content = base64.b64decode(inline["data"], validate=True)
                if not 0 < len(content) <= 16 * 1024 ** 2:
                    raise ProviderFailure("image-output-size-limit")
                with Image.open(BytesIO(content)) as opened:
                    if opened.format != formats[mime][0] or min(opened.size) < 16 or max(opened.size) > 2048 or getattr(opened, "n_frames", 1) != 1:
                        raise ProviderFailure("invalid-image-output")
                    opened.verify()
            except (KeyError, TypeError, OSError, ValueError, Image.DecompressionBombError):
                raise ProviderFailure("invalid-image-output") from None
            images.append({"content": content, "extension": formats[mime][1], "mime_type": mime,
                           "sha256": hashlib.sha256(content).hexdigest()})
    for choice in response.get("choices", []):
        value = choice.get("message", {}).get("content")
        if isinstance(value, str):
            texts.append(value)
    return texts, images


def generate(budget, keys, request_id, model, prompt, output_tokens=2048, references=()):
    budget.require_open()
    quote(model, output_tokens)
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 32_768 or len(references) > 2:
        raise BudgetRefused("Use a bounded prompt and at most two staged reference crops")
    policy = MODELS[model]
    provider = policy["provider"]
    key = keys.get(PROVIDER_KEYS[provider])
    if not key:
        raise ProviderFailure("vault-key-missing")
    input_tokens = None
    if provider == "gemini":
        parts = [{"text": prompt}, *(image_part(raw) for raw in references)]
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"candidateCount": 1, "maxOutputTokens": output_tokens}}
        if policy["image_output"]:
            payload["generationConfig"].update({"responseModalities": ["IMAGE", "TEXT"], "imageConfig": {"imageSize": "1K", "aspectRatio": "1:1"}})
        count = request_json(provider, key, "models/" + model + ":countTokens", {"contents": payload["contents"]})
        input_tokens = count.get("totalTokens")
        if type(input_tokens) is not int or input_tokens <= 0:
            raise ProviderFailure("token-count-missing")
        action = "models/" + model + ":generateContent"
    elif provider == "kimi":
        content = [{"type": "text", "text": prompt}]
        for raw in references:
            encoded = image_part(raw)["inlineData"]["data"]
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        messages = [{"role": "user", "content": content if references else prompt}]
        count = request_json(provider, key, "tokenizers/estimate-token-count", {"model": model, "messages": messages})
        estimated = count.get("data", {}).get("total_tokens")
        if type(estimated) is not int or estimated <= 0:
            raise ProviderFailure("token-count-missing")
        input_tokens = estimated + 4096
        payload = {"model": model, "messages": messages, "max_completion_tokens": output_tokens,
                   "stream": False, "reasoning_effort": "low"}
        action = "chat/completions"
    else:
        if references and model != "glm-5.3-flash":
            raise BudgetRefused("This Z.ai adapter is text-only, not a vision or image model")
        content = [{"type": "text", "text": prompt}]
        for raw in references:
            encoded = image_part(raw)["inlineData"]["data"]
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {"model": model, "messages": [{"role": "user", "content": content if references else prompt}],
                   "max_tokens": output_tokens, "stream": False, "thinking": {"type": "enabled"}, "reasoning_effort": "low"}
        action = "chat/completions"
    amount, limits = quote(model, output_tokens, input_tokens)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    limits["reference_sha256"] = [hashlib.sha256(raw).hexdigest() for raw in references]
    budget.reserve(request_id, provider, model, amount, digest, limits)
    started = time.monotonic()
    try:
        response = request_json(provider, key, action, payload)
        usage = response.get("usageMetadata", response.get("usage", {}))
        safe_usage = {name: value for name, value in usage.items() if type(value) is int and value >= 0} if isinstance(usage, dict) else {}
        outcome = {"usage": safe_usage, "elapsed_seconds": time.monotonic() - started,
                   "response_model": response.get("modelVersion", response.get("model")), "response_sha256": hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()}
        budget.finish(request_id, "returned", outcome)
        return {"request_id": request_id, "reserved_microusd": amount, "outcome": outcome, "response": response}
    except BaseException as error:
        budget.finish(request_id, "failed-unknown-charge", {"error": error.category if isinstance(error, ProviderFailure) else "interrupted-or-invalid-response", "elapsed_seconds": time.monotonic() - started})
        raise
