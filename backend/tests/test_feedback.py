from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feedback  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    feedback.configure_storage(tmp_path / "feedback")
    app = FastAPI()
    app.include_router(feedback.router)
    return TestClient(app)


def _create_item(client: TestClient, **overrides):
    payload = {
        "title": "Viewer panel breaks",
        "body": "The viewer panel disappears after changing tabs.",
        "type": "Bug",
        "priority": "High",
        "status": "New",
        "page_url": "https://splat.lab/view?scene=splat_abc&token=secret-token&ok=yes#frag",
        "page_path": "/view",
        "page_tab": "inspect",
        "component_label": "Splat viewer",
        "tags_json": ["viewer", "regression", "viewer"],
        "context_json": {
            "route": {
                "url": "https://splat.lab/view?session=secret-session&scene=splat_abc",
                "path": "/view?api_key=secret-key&scene=splat_abc",
            },
            "authorization": "Bearer never-store",
            "cookies": {"session": "never-store"},
            "request_body": {"password": "never-store"},
            "recent_failed_api_calls": [
                {
                    "method": "post",
                    "url": "/api/splat/jobs?token=never-store&scene=splat_abc",
                    "status": "500",
                    "duration_ms": "42",
                    "request_body": {"raw": "never-store"},
                    "response_body": "never-store",
                }
            ],
        },
        "resolution_notes": "",
        "resolution_metadata_json": {},
        "created_by": "tester",
    }
    payload.update(overrides)
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_detail_and_sanitization(client: TestClient):
    item = _create_item(client)

    assert item["title"] == "Viewer panel breaks"
    assert item["type"] == "Bug"
    assert item["tags_json"] == ["viewer", "regression"]
    assert item["page_url"] == "https://splat.lab/view?scene=splat_abc&ok=yes"

    serialized = json.dumps(item)
    assert "secret-token" not in serialized
    assert "secret-session" not in serialized
    assert "secret-key" not in serialized
    assert "never-store" not in serialized

    failed_call = item["context_json"]["recent_failed_api_calls"][0]
    assert failed_call == {
        "method": "POST",
        "path": "/api/splat/jobs?scene=splat_abc",
        "status": 500,
        "duration_ms": 42,
    }

    detail = client.get(f"/api/feedback/{item['id']}").json()
    assert detail["comments"] == []
    assert detail["attachments"] == []
    assert detail["created_at"]
    assert detail["updated_at"]


def test_queue_filters_status_patch_and_search(client: TestClient):
    new_item = _create_item(client, title="Active viewer issue", component_label="Viewer")
    fixed_item = _create_item(client, title="Ready for user test", component_label="Verifier")
    archived_item = _create_item(client, title="Old archived note", component_label="Archive")

    fixed = client.patch(f"/api/feedback/{fixed_item['id']}", json={"status": "Fixed"}).json()
    archived = client.patch(f"/api/feedback/{archived_item['id']}", json={"status": "Archived"}).json()

    assert fixed["completed_at"] is None
    assert fixed["archived_at"] is None
    assert archived["completed_at"] is not None
    assert archived["archived_at"] is not None

    active_ids = {item["id"] for item in client.get("/api/feedback").json()["items"]}
    assert new_item["id"] in active_ids
    assert fixed_item["id"] in active_ids
    assert archived_item["id"] not in active_ids

    user_ids = {item["id"] for item in client.get("/api/feedback", params={"queue": "user"}).json()["items"]}
    assert user_ids == {fixed_item["id"]}

    terminal_ids = {
        item["id"] for item in client.get("/api/feedback", params={"queue": "terminal"}).json()["items"]
    }
    assert terminal_ids == {archived_item["id"]}

    search_ids = {
        item["id"] for item in client.get("/api/feedback", params={"queue": "all", "search": "verifier"}).json()["items"]
    }
    assert search_ids == {fixed_item["id"]}


def test_comments_are_chronological(client: TestClient):
    item = _create_item(client)

    first = client.post(
        f"/api/feedback/{item['id']}/comments",
        json={"body": "I can reproduce this.", "created_by": "tester"},
    )
    second = client.post(
        f"/api/feedback/{item['id']}/comments",
        json={"body": "Started a backend fix.", "created_by": "codex"},
    )
    assert first.status_code == 201
    assert second.status_code == 201

    comments = client.get(f"/api/feedback/{item['id']}/comments").json()["items"]
    assert [comment["body"] for comment in comments] == [
        "I can reproduce this.",
        "Started a backend fix.",
    ]


def test_multiple_attachments_can_be_listed_and_streamed(client: TestClient):
    item = _create_item(client)

    response = client.post(
        f"/api/feedback/{item['id']}/attachments",
        data={"created_by": "tester"},
        files=[
            ("files", ("screen shot.png", b"fake-image-bytes", "image/png")),
            ("files", ("notes.txt", b"plain notes", "text/plain")),
        ],
    )
    assert response.status_code == 201, response.text
    attachments = response.json()["items"]
    assert len(attachments) == 2
    assert attachments[0]["original_name"] == "screen shot.png"
    assert attachments[0]["size_bytes"] == len(b"fake-image-bytes")

    listed = client.get(f"/api/feedback/{item['id']}/attachments").json()["items"]
    assert [attachment["original_name"] for attachment in listed] == ["screen shot.png", "notes.txt"]

    streamed = client.get(f"/api/feedback/{item['id']}/attachments/{attachments[0]['id']}/stream")
    assert streamed.status_code == 200
    assert streamed.content == b"fake-image-bytes"
    assert streamed.headers["content-type"].startswith("image/png")


def test_app_context_is_safe_and_available(client: TestClient):
    context = client.get("/api/app-context").json()

    assert context["service"] == "splatlab"
    assert context["feedback_context_version"] == 1
    assert "git_short_commit" in context
