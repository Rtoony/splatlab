"""Auth rails on a publicly tunneled app with one shared secret.

PORTAL_TOKEN is simultaneously the password and the HMAC signing key, so the
login endpoint is the single point where that secret can be guessed. Three gaps
are covered here: an unthrottled POST /login, a signed cookie whose timestamp
was only checked in one direction, and the SPA catch-all answering unknown
/api/* GETs with HTML 200.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as splat_main  # noqa: E402

TOKEN = "test-portal-token-0123456789"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    monkeypatch.setattr(splat_main, "_LOGIN_ATTEMPTS", {})
    return TestClient(splat_main.app, follow_redirects=False)


# ---------------------------------------------------------------------------
# Login throttle
# ---------------------------------------------------------------------------

def test_a_correct_token_logs_in(client):
    r = client.post("/login", data={"portal_token": TOKEN})
    assert r.status_code == 303
    assert splat_main.COOKIE in r.cookies


def test_a_wrong_token_is_rejected(client):
    assert client.post("/login", data={"portal_token": "nope"}).status_code == 401


def test_repeated_failures_are_throttled(client):
    for _ in range(splat_main._LOGIN_MAX_ATTEMPTS):
        assert client.post("/login", data={"portal_token": "nope"}).status_code == 401

    r = client.post("/login", data={"portal_token": "nope"})
    assert r.status_code == 429
    assert r.headers["retry-after"]


def test_the_throttle_blocks_even_the_correct_token(client):
    """Otherwise an attacker's guesses are only slowed once they succeed."""
    for _ in range(splat_main._LOGIN_MAX_ATTEMPTS):
        client.post("/login", data={"portal_token": "nope"})

    assert client.post("/login", data={"portal_token": TOKEN}).status_code == 429


def test_a_success_clears_the_bucket(client):
    for _ in range(splat_main._LOGIN_MAX_ATTEMPTS - 1):
        client.post("/login", data={"portal_token": "nope"})

    assert client.post("/login", data={"portal_token": TOKEN}).status_code == 303
    assert client.post("/login", data={"portal_token": "nope"}).status_code == 401


def test_attempts_age_out_of_the_window(client, monkeypatch):
    for _ in range(splat_main._LOGIN_MAX_ATTEMPTS):
        client.post("/login", data={"portal_token": "nope"})
    assert client.post("/login", data={"portal_token": "nope"}).status_code == 429

    real_time = time.time
    monkeypatch.setattr(splat_main.time, "time",
                        lambda: real_time() + splat_main._LOGIN_WINDOW_S + 1)

    assert client.post("/login", data={"portal_token": "nope"}).status_code == 401


def test_throttling_is_per_client(client):
    for _ in range(splat_main._LOGIN_MAX_ATTEMPTS):
        client.post("/login", data={"portal_token": "nope"},
                    headers={"x-forwarded-for": "10.0.0.1"})

    assert client.post("/login", data={"portal_token": "nope"},
                       headers={"x-forwarded-for": "10.0.0.1"}).status_code == 429
    assert client.post("/login", data={"portal_token": "nope"},
                       headers={"x-forwarded-for": "10.0.0.2"}).status_code == 401


# ---------------------------------------------------------------------------
# Cookie validity
# ---------------------------------------------------------------------------

def test_a_fresh_cookie_is_valid(monkeypatch):
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    assert splat_main._valid_cookie(splat_main._sign(int(time.time()))) is True


def test_an_expired_cookie_is_rejected(monkeypatch):
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    stale = int(time.time()) - splat_main.MAX_AGE - 60
    assert splat_main._valid_cookie(splat_main._sign(stale)) is False


def test_a_future_dated_cookie_is_rejected(monkeypatch):
    """Only `age > MAX_AGE` was checked, so a cookie stamped years ahead never
    expired — a validly-signed forever-session."""
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    far_future = int(time.time()) + 10 * 365 * 24 * 3600
    assert splat_main._valid_cookie(splat_main._sign(far_future)) is False


def test_small_clock_skew_is_tolerated(monkeypatch):
    """A little skew must not log everyone out."""
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    slightly_ahead = int(time.time()) + 30
    assert splat_main._valid_cookie(splat_main._sign(slightly_ahead)) is True


def test_a_tampered_mac_is_rejected(monkeypatch):
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    ts_str, mac = splat_main._sign(int(time.time())).rsplit(":", 1)
    forged = f"{ts_str}:{'0' * len(mac)}"
    assert splat_main._valid_cookie(forged) is False


def test_a_cookie_for_a_different_timestamp_is_rejected(monkeypatch):
    """The MAC must cover the timestamp it is presented with."""
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    _, mac = splat_main._sign(int(time.time())).rsplit(":", 1)
    assert splat_main._valid_cookie(f"{int(time.time()) + 1}:{mac}") is False


@pytest.mark.parametrize("junk", ["", "no-colon", ":", "abc:def", "nan:x"])
def test_malformed_cookies_are_rejected(monkeypatch, junk):
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    assert splat_main._valid_cookie(junk) is False


# ---------------------------------------------------------------------------
# Unknown API routes
# ---------------------------------------------------------------------------

def test_an_unknown_api_get_is_a_json_404_not_the_spa(client):
    """This used to return index.html with a 200, so a typo'd or removed
    endpoint looked like a working route serving HTML."""
    r = client.get("/api/splat/does-not-exist",
                   headers={"authorization": f"Bearer {TOKEN}"})

    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert "<html" not in r.text.lower()


def test_an_unknown_api_route_404s_even_unauthenticated(client):
    """An API path must not be answered with a login redirect either."""
    assert client.get("/api/splat/does-not-exist").status_code == 404


def test_a_normal_spa_path_still_redirects_when_unauthenticated(client):
    r = client.get("/view/splat_abc")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_healthz_is_still_open(client):
    assert client.get("/healthz").status_code == 200
