from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

import cloud_budget
from cloud_budget import BudgetRefused, CloudBudget
import cloud_experiments as cloud


@pytest.fixture
def budget(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_budget.time, "time", lambda: 1_788_761_000)
    result = CloudBudget(tmp_path / "budget.sqlite3")
    result.initialize(20_000_000, 1_788_780_000)
    return result


def reserve(budget, name, amount=1_000_000):
    budget.reserve(name, "gemini", "gemini-3.8-flash", amount, "a" * 64, {"max_output_tokens": 512})


def test_authorization_is_immutable_and_restart_retains_all_reservations(budget):
    reserve(budget, "first")
    restarted = CloudBudget(budget.path)
    assert restarted.initialize(20_000_000, 1_788_780_000)["held_microusd"] == 1_000_000
    for amount, deadline in ((21_000_000, 1_788_780_000), (20_000_000, 1_788_781_000), (19_000_000, 1_788_780_000)):
        with pytest.raises(BudgetRefused):
            restarted.initialize(amount, deadline)
    assert restarted.status()["held_microusd"] == 1_000_000


def test_missing_authorization_does_not_implicitly_create_budget(tmp_path):
    path = tmp_path / "absent.sqlite3"
    with pytest.raises(cloud_budget.sqlite3.OperationalError):
        CloudBudget(path).status()
    assert not path.exists()


def test_new_fifty_dollar_authorization_retains_a_single_aggregate_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_budget.time, "time", lambda: 1_788_761_000)
    ledger = CloudBudget(tmp_path / "new-authorization.sqlite3")
    ledger.initialize(50_000_000, 1_788_780_000)
    reserve(ledger, "first", 49_000_000)
    reserve(ledger, "second", 1_000_000)
    with pytest.raises(BudgetRefused, match="Aggregate"):
        reserve(ledger, "over-cap", 1)
    assert CloudBudget(ledger.path).status()["available_microusd"] == 0


@pytest.mark.parametrize("amount", [0, -1, True, 50_000_001])
def test_invalid_new_authorization_refuses_before_creating_ledger(tmp_path, amount):
    ledger = CloudBudget(tmp_path / "invalid-authorization.sqlite3")
    with pytest.raises(BudgetRefused, match="Explicit cloud cap"):
        ledger.initialize(amount, int(cloud_budget.time.time()) + 600)
    assert not ledger.path.exists()


def test_aggregate_cap_is_atomic_across_concurrent_requests(budget):
    def attempt(number):
        try:
            reserve(CloudBudget(budget.path), f"concurrent-{number}", 3_000_000)
            return True
        except BudgetRefused:
            return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(attempt, range(20))) == 6
    assert budget.status()["held_microusd"] == 18_000_000
    assert budget.status()["available_microusd"] == 2_000_000


@pytest.mark.parametrize("amount", [0, -1, 0.01, True, 20_000_001])
def test_bad_and_over_cap_amounts_cannot_admit(budget, amount):
    with pytest.raises(BudgetRefused):
        reserve(budget, "invalid", amount)
    assert not budget.status()["requests"]


def test_duplicate_request_and_unknown_failure_never_release_money(budget):
    reserve(budget, "uncertain", 20_000_000)
    budget.finish("uncertain", "failed-unknown-charge", {"error": "timeout"})
    for name in ("uncertain", "new-attempt"):
        with pytest.raises(BudgetRefused):
            reserve(budget, name, 1)
    with pytest.raises(BudgetRefused):
        budget.finish("uncertain", "returned", {})
    assert budget.status()["available_microusd"] == 0


def test_expiry_and_cleanup_margin_refuse_new_work(budget, monkeypatch):
    monkeypatch.setattr(cloud_budget.time, "time", lambda: 1_788_779_761)
    with pytest.raises(BudgetRefused, match="window"):
        reserve(budget, "late")
    with pytest.raises(BudgetRefused, match="window"):
        budget.require_open()


def test_symlinked_ledger_refuses(tmp_path):
    path = tmp_path / "alias"
    path.symlink_to(tmp_path / "actual")
    with pytest.raises(BudgetRefused, match="symlinked"):
        CloudBudget(path)


def test_rate_card_uses_integer_microdollars_and_highest_image_rate(budget):
    amount, limits = cloud.quote("gemini-3.1-flash-image", 8192, 4000)
    assert amount == 1_007_040
    assert limits["output_usd_per_million"] == "60"
    assert limits["automatic_retries"] == 0
    assert not limits["tools_enabled"]
    with pytest.raises(BudgetRefused):
        cloud.quote("unpriced-model", 512)
    with pytest.raises(BudgetRefused):
        cloud.quote("glm-5.3", 8193)


def test_expired_price_card_refuses(budget, monkeypatch):
    monkeypatch.setattr(cloud.time, "time", lambda: cloud.PRICING_EXPIRES_UTC)
    with pytest.raises(BudgetRefused, match="expired"):
        cloud.quote("gemini-3.8-flash", 512)


def test_vault_reads_only_allowed_key_fields_and_never_evaluates_exports(monkeypatch):
    output = "export GEMINI_API_KEY=fixture-key\nexport DATABASE_URL=unrelated-private-value\nexport ZAI_API_KEY=$(touch /tmp/never)\nexport NEXUS_INJECT_STATUS=ok\n"
    monkeypatch.setattr(cloud.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output, returncode=0))
    with pytest.raises(cloud.ProviderFailure, match="format") as caught:
        cloud.vault_keys()
    assert "fixture-key" not in str(caught.value)
    output = "export GEMINI_API_KEY=fixture-key\nexport DATABASE_URL=unrelated-private-value\nexport NEXUS_INJECT_STATUS=ok\n"
    assert cloud.vault_keys() == {"GEMINI_API_KEY": "fixture-key"}


def test_missing_vault_session_is_an_actionable_redacted_error(monkeypatch):
    monkeypatch.setattr(cloud.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="export NEXUS_INJECT_STATUS=no-session\n", returncode=0))
    with pytest.raises(cloud.ProviderFailure, match="nexus-unlock"):
        cloud.vault_keys()


def test_provider_endpoint_allowlist_and_redirects_refuse():
    with pytest.raises(cloud.ProviderFailure, match="endpoint"):
        cloud.request_json("gemini", "fixture-secret", "https://untrusted.example")
    with pytest.raises(cloud.ProviderFailure, match="redirect"):
        cloud.NoRedirect().redirect_request(None, None, 302, "", {}, "https://untrusted.example")


def test_gemini_counts_free_input_then_reserves_before_paid_call(budget, monkeypatch):
    calls = []
    def request(provider, key, action, payload=None):
        calls.append(action)
        if action.endswith(":countTokens"):
            assert not budget.status()["requests"]
            return {"totalTokens": 20}
        assert budget.status()["requests"][0]["status"] == "reserved"
        assert payload["generationConfig"] == {"candidateCount": 1, "maxOutputTokens": 512}
        assert "tools" not in payload
        return {"modelVersion": "gemini-3.8-flash", "usageMetadata": {"promptTokenCount": 20, "totalTokenCount": 25}}
    monkeypatch.setattr(cloud, "request_json", request)
    result = cloud.generate(budget, {"GEMINI_API_KEY": "fixture-secret"}, "gemini-smoke", "gemini-3.8-flash", "Return OK", 512)
    assert len(calls) == 2
    assert budget.status()["requests"][0]["status"] == "returned"
    assert "fixture-secret" not in json.dumps(budget.status())
    assert result["outcome"]["usage"]["totalTokenCount"] == 25


def test_token_count_over_model_limit_refuses_paid_call(budget, monkeypatch):
    calls = []
    def request(provider, key, action, payload=None):
        calls.append(action)
        return {"totalTokens": 2_000_000}
    monkeypatch.setattr(cloud, "request_json", request)
    with pytest.raises(BudgetRefused):
        cloud.generate(budget, {"GEMINI_API_KEY": "fixture"}, "oversized", "gemini-3.8-flash", "Return OK", 512)
    assert len(calls) == 1
    assert not budget.status()["requests"]


def test_transport_failure_keeps_full_hold_and_never_retries(budget, monkeypatch):
    calls = []
    def request(provider, key, action, payload=None):
        calls.append(action)
        assert budget.status()["held_microusd"] > 0
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "low"
        raise cloud.ProviderFailure("transport-or-response-error")
    monkeypatch.setattr(cloud, "request_json", request)
    with pytest.raises(cloud.ProviderFailure):
        cloud.generate(budget, {"ZAI_API_KEY": "fixture"}, "zai-smoke", "glm-5.3", "Return OK", 512)
    assert len(calls) == 1
    assert budget.status()["requests"][0]["status"] == "failed-unknown-charge"
    assert budget.status()["held_microusd"] > 0


def test_api_key_absence_never_reserves_or_contacts_provider(budget, monkeypatch):
    monkeypatch.setattr(cloud, "request_json", lambda *args, **kwargs: pytest.fail("Network must not run"))
    with pytest.raises(cloud.ProviderFailure, match="key-missing"):
        cloud.generate(budget, {}, "no-key", "gemini-3.8-flash", "Return OK")
    assert not budget.status()["requests"]


def test_png_reference_is_validated_and_sha_pinned(budget, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    raw = buffer.getvalue()
    def request(provider, key, action, payload=None):
        if action.endswith(":countTokens"):
            return {"totalTokens": 500}
        assert payload["generationConfig"]["imageConfig"]["imageSize"] == "1K"
        assert payload["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"
        assert cloud.base64.b64decode(payload["contents"][0]["parts"][1]["inlineData"]["data"]) == raw
        return {"usageMetadata": {"totalTokenCount": 1000}}
    monkeypatch.setattr(cloud, "request_json", request)
    cloud.generate(budget, {"GEMINI_API_KEY": "fixture"}, "image-test", "gemini-3.1-flash-image", "Synthetic fixture", 8192, [raw])
    assert budget.status()["requests"][0]["limits"]["reference_sha256"] == [hashlib.sha256(raw).hexdigest()]
    with pytest.raises(BudgetRefused):
        cloud.image_part(b"not an image")


def test_kimi_uses_current_completion_cap_and_token_estimate_margin(budget, monkeypatch):
    calls = []
    def request(provider, key, action, payload=None):
        calls.append(action)
        assert provider == "kimi"
        if action == "tokenizers/estimate-token-count":
            assert not budget.status()["requests"]
            return {"data": {"total_tokens": 241}}
        assert payload["max_completion_tokens"] == 2048
        assert "max_tokens" not in payload
        assert "thinking" not in payload
        assert payload["reasoning_effort"] == "low"
        assert "tools" not in payload
        assert budget.status()["requests"][0]["limits"]["input_tokens_bound"] == 4337
        return {"model": "kimi-k3", "usage": {"prompt_tokens": 241, "completion_tokens": 512}}
    monkeypatch.setattr(cloud, "request_json", request)
    cloud.generate(budget, {"KIMI_API_KEY": "fixture"}, "kimi-smoke", "kimi-k3", "Return OK", 2048)
    assert len(calls) == 2
    assert budget.status()["held_microusd"] == 107462


def test_pro_image_price_is_separate_from_flash_image(budget):
    amount, limits = cloud.quote("gemini-3-pro-image", 8192, 4000)
    assert amount == 2_002_080
    assert limits["output_usd_per_million"] == "120"


def test_only_multimodal_glm_flash_accepts_reference_crops(budget, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    raw = buffer.getvalue()
    def request(provider, key, action, payload=None):
        assert provider == "zai"
        assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        return {"model": "glm-5.3-flash", "usage": {"prompt_tokens": 1000, "completion_tokens": 512}}
    monkeypatch.setattr(cloud, "request_json", request)
    with pytest.raises(BudgetRefused, match="text-only"):
        cloud.generate(budget, {"ZAI_API_KEY": "fixture"}, "glm-not-vision", "glm-5.3", "Inspect fixture", references=[raw])
    cloud.generate(budget, {"ZAI_API_KEY": "fixture"}, "glm-vision", "glm-5.3-flash", "Inspect fixture", references=[raw])
    assert len(budget.status()["requests"]) == 1


@pytest.mark.parametrize("image_format,mime,extension", [("PNG", "image/png", "png"), ("JPEG", "image/jpeg", "jpg")])
def test_generated_image_formats_preserve_original_bytes(image_format, mime, extension):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format=image_format)
    raw = buffer.getvalue()
    inline = {"inlineData": {"mimeType": mime, "data": cloud.base64.b64encode(raw).decode()}}
    response = {"candidates": [{"content": {"parts": [{"text": "private reasoning", "thought": True}, {"text": "Candidate"}, inline]}}]}
    texts, images = cloud.response_assets(response)
    assert texts == ["Candidate"]
    assert images == [{"content": raw, "mime_type": mime, "extension": extension, "sha256": hashlib.sha256(raw).hexdigest()}]


@pytest.mark.parametrize("mime,data", [("image/webp", "AAAA"), ("image/png", "not base64"), ("image/jpeg", "AAAA")])
def test_invalid_generated_images_refuse_instead_of_silently_disappearing(mime, data):
    with pytest.raises(cloud.ProviderFailure):
        cloud.response_assets({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]}}]})
