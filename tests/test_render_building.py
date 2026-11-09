"""/render with the OpenAI HTTP calls patched: the request carries the depth or preview image and the building prompt.
No network call, no API key (a dummy string is passed and never leaves the process)."""
import base64
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api import server  # noqa: E402
from packages.floorplan_core import render_backend as RB  # noqa: E402

PNG_A = b"\x89PNG\r\n\x1a\n" + b"depth-bytes"
PNG_B = b"\x89PNG\r\n\x1a\n" + b"preview-bytes"
OUT = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"stub").decode()
PROMPT = ("A photorealistic exterior architectural photograph of a 4-storey residential apartment building, "
          "14.4 m tall. The ground floor holds the entrance lobby. Punched window openings sit in a regular rhythm "
          "on every storey; living rooms have large windows with a transom band. View: birds-eye exterior oblique view.")


def url(b):
    return "data:image/png;base64," + base64.b64encode(b).decode()


def run(monkeypatch, body):
    calls = []

    def fake_multipart(u, fields, files, timeout=30, extra_headers=None):
        calls.append((u, dict(fields), files))
        return 200, {"data": [{"b64_json": base64.b64encode(b"stub").decode()}]}

    monkeypatch.setattr(RB, "http_post_multipart", fake_multipart)
    monkeypatch.setattr(RB, "http_post_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fallback call expected")))
    monkeypatch.setattr(RB, "_extract_openai_image_data_url", lambda payload: OUT)
    r = server.app.test_client().post("/render", json=body)
    return r, calls


def test_render_depth_only_sends_depth_and_building_prompt(monkeypatch):
    r, calls = run(monkeypatch, {"backend": "openai", "prompt": PROMPT, "depth_data_url": url(PNG_A),
                                 "openai_api_key": "dummy-not-a-key", "openai_width": 1024, "openai_height": 1024})
    assert r.status_code == 200 and r.get_json()["image_data_url"] == OUT
    assert len(calls) == 1
    u, fields, files = calls[0]
    assert u.endswith("/images/edits")
    assert files["image"]["content"] == PNG_A
    assert "4-storey residential apartment building" in fields["prompt"]
    assert "transom" in fields["prompt"] and "entrance lobby" in fields["prompt"]


def test_render_with_preview_uses_preview_and_still_receives_depth(monkeypatch):
    r, calls = run(monkeypatch, {"backend": "openai", "prompt": PROMPT, "depth_data_url": url(PNG_A),
                                 "preview_data_url": url(PNG_B), "openai_api_key": "dummy-not-a-key"})
    assert r.status_code == 200
    assert calls[0][2]["image"]["content"] == PNG_B
    assert r.get_json()["meta"]["reference_type"] == "preview"


def test_render_without_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r, calls = run(monkeypatch, {"backend": "openai", "prompt": PROMPT, "depth_data_url": url(PNG_A)})
    assert r.status_code == 500 and "API key" in r.get_json()["error"] and not calls


def test_render_requires_depth(monkeypatch):
    r, calls = run(monkeypatch, {"backend": "openai", "prompt": PROMPT})
    assert r.status_code == 400 and not calls


def test_edits_failure_raises_and_never_calls_generations(monkeypatch):
    gen_calls = []

    def boom(u, fields, files, timeout=30, extra_headers=None):
        raise RuntimeError("HTTP 403 POST https://api.openai.com/v1/images/edits: model not allowed")

    monkeypatch.setattr(RB, "http_post_multipart", boom)
    monkeypatch.setattr(RB, "http_post_json", lambda *a, **k: gen_calls.append(a) or (200, {}))
    with pytest.raises(RuntimeError) as ei:
        RB._render_with_openai(PROMPT, url(PNG_A), api_key="dummy-not-a-key")
    assert "HTTP 403" in str(ei.value) and "model not allowed" in str(ei.value)
    assert not gen_calls
    r = server.app.test_client().post("/render", json={"backend": "openai", "prompt": PROMPT,
                                                       "depth_data_url": url(PNG_A), "openai_api_key": "dummy-not-a-key"})
    assert r.status_code == 500 and "HTTP 403" in r.get_json()["error"] and not gen_calls


def test_no_reference_raises_clear_error(monkeypatch):
    calls = []
    monkeypatch.setattr(RB, "http_post_multipart", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(RB, "http_post_json", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="capture maps first"):
        RB._render_with_openai(PROMPT, None, preview_data_url=None, api_key="dummy-not-a-key")
    assert not calls


def test_default_model_is_new(monkeypatch):
    assert RB.DEFAULT_OPENAI_MODEL == "gpt-image-2.5-flare"
    assert server.DEFAULT_OPENAI_MODEL == "gpt-image-2.5-flare"
    r, calls = run(monkeypatch, {"backend": "openai", "prompt": PROMPT, "depth_data_url": url(PNG_A),
                                 "openai_api_key": "dummy-not-a-key"})
    assert r.status_code == 200
    assert calls[0][1]["model"] == "gpt-image-2.5-flare"
    meta = r.get_json()["meta"]
    assert meta["model"] == "gpt-image-2.5-flare" and meta["endpoint"] == "edits"
