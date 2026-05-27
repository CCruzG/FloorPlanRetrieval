import json
import uuid
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


def http_post_json(url, payload, timeout=30, extra_headers=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    req = urlrequest.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} POST {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc


def http_get_json(url, timeout=30):
    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} GET {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def http_get_bytes(url, params=None, timeout=30):
    full_url = url
    if params:
        full_url = f"{url}?{urlparse.urlencode(params)}"
    req = urlrequest.Request(full_url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} GET {full_url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"GET {full_url} failed: {exc.reason}") from exc


def http_post_multipart(url, fields, files, timeout=30, extra_headers=None):
    boundary = f"----fp-boundary-{uuid.uuid4().hex}"
    body = bytearray()

    for key, val in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(val).encode("utf-8"))
        body.extend(b"\r\n")

    for key, file_info in (files or {}).items():
        filename = file_info.get("filename", "upload.bin")
        content_type = file_info.get("content_type", "application/octet-stream")
        content = file_info.get("content", b"")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    req = urlrequest.Request(
        url,
        data=bytes(body),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} POST {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc
