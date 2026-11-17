import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(built):
    from app.main import app

    with TestClient(app) as c:
        yield c


def jpeg_bytes(color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color).save(buf, "JPEG")
    return buf.getvalue()


def test_health_and_meta(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["items"] == 12
    m = client.get("/api/meta").json()
    assert m["stations"] and m["trains"] and "umbrella" in m["categories"]


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Khoya" in r.text


def test_text_search_understands_query(client):
    r = client.post("/api/search", data={"description": "lost my red backpack on 12951"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["understood"]["train_no"] == "12951"
    assert body["understood"]["looking_for"] == "a photo of a red backpack"
    assert 1 <= len(body["results"]) <= 8
    first = body["results"][0]
    assert first["strength"] in {"strong", "possible", "weak"} and first["image_url"].startswith("/images/")


def test_photo_search_and_combined(client):
    r = client.post("/api/search", files={"photo": ("bag.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["understood"]["used_photo"] is True
    r = client.post("/api/search", data={"description": "blue umbrella"},
                    files={"photo": ("u.jpg", jpeg_bytes((20, 20, 220)), "image/jpeg")})
    assert r.status_code == 200 and r.json()["results"]


def test_image_served(client):
    item = client.post("/api/search", data={"description": "wallet"}).json()["results"][0]
    r = client.get(item["image_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/images/NOPE").status_code == 404


@pytest.mark.parametrize("kwargs,code", [
    ({"data": {}}, 400),                                                    # nothing to search with
    ({"data": {"description": "bag", "lost_date": "11/11/2026"}}, 400),     # bad date format
    ({"files": {"photo": ("x.jpg", b"not an image", "image/jpeg")}}, 400),  # corrupt upload
])
def test_bad_requests(client, kwargs, code):
    assert client.post("/api/search", **kwargs).status_code == code


def test_upload_size_limit(client):
    big = b"\xff" * (7 * 1024 * 1024)
    r = client.post("/api/search", files={"photo": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_feedback_and_metrics(client):
    item = client.post("/api/search", data={"description": "suitcase"}).json()["results"][0]
    ok = client.post("/api/feedback", json={"query_id": "abc", "item_id": item["item_id"], "rank": 1})
    assert ok.status_code == 200
    assert client.post("/api/feedback", json={"query_id": "abc", "item_id": "NOPE", "rank": 1}).status_code == 404
    text = client.get("/metrics").text
    assert "khoya_search_requests_total" in text and "khoya_feedback_total" in text
