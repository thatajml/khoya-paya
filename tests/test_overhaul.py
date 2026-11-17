import html.parser
import subprocess
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client(built):
    with TestClient(app) as c:
        yield c


class DOMCollector(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements_by_id = {}
        self.elements_by_tag = []
        self.labels_for = set()

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        self.elements_by_tag.append((tag, attr_dict))
        if "id" in attr_dict:
            self.elements_by_id[attr_dict["id"]] = (tag, attr_dict)
        if tag == "label" and "for" in attr_dict:
            self.labels_for.add(attr_dict["for"])


def test_git_branch_is_v1():
    """Verify that current git branch is strictly v1 (Requirement R1)."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    assert branch == "v1", f"Expected active git branch to be 'v1', got '{branch}'"


def test_home_serves_beige_palette_and_design_tokens(client):
    """Verify GET / serves HTML with the complete warm paper/espresso ink palette."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    text = r.text

    # Required Beige Design System Color Tokens
    required_tokens = {
        "#F4EDE1": "Paper (page background)",
        "#FBF7F0": "Card (raised card surfaces)",
        "#EBE1CF": "Sand (chips, banner, wells)",
        "#D8CAB2": "Linen (borders, dividers)",
        "#8C7A60": "Field (input borders, icons)",
        "#2A241D": "Ink (primary text)",
        "#5F5446": "Muted (secondary/helper text)",
        "#3B2F24": "Espresso (primary action button)",
        "#9A4320": "Rust (focus ring, accent dot, links)",
        "#4F5B26": "Olive (strong match badge and meter)",
        "#8A6A3A": "Amber (good match badge and meter)",
        "#E4E6CF": "Sage (claimed state confirmation box)",
        "#FFFDF9": "Input well background",
    }
    for hex_code, desc in required_tokens.items():
        assert hex_code.lower() in text.lower(), f"Missing design token {hex_code} ({desc})"


def test_home_serves_typography_and_bilingual_wordmark(client):
    """Verify typography font linkages and bilingual Hindi/English wordmark."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text

    # Google Fonts linkages
    assert "fonts.googleapis.com" in text
    assert "Fraunces" in text, "Display serif font Fraunces must be included"
    assert "Inter" in text, "UI body font Inter must be included"
    assert "Noto+Serif+Devanagari" in text or "Noto Serif Devanagari" in text, (
        "Hindi wordmark font Noto Serif Devanagari must be included"
    )

    # Bilingual Wordmark
    assert "Khoya" in text and "Paya" in text, "English wordmark missing"
    assert "·" in text, "Dot separator in Khoya·Paya missing"
    assert "खोया-पाया" in text, "Hindi wordmark 'खोया-पाया' missing"


def test_prototype_banner_and_disclaimer(client):
    """Verify top disclaimer banner stating records are synthetic demo data."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text

    assert "Prototype" in text
    assert "synthetic demo data" in text
    assert "Wikimedia Commons" in text
    assert "not an official Indian Railways service" in text


def test_required_dom_ids_and_form_bindings(client):
    """Verify all mandatory DOM IDs and form control names are preserved."""
    r = client.get("/")
    assert r.status_code == 200

    collector = DOMCollector()
    collector.feed(r.text)

    # All required DOM IDs from specification and legacy interface
    required_ids = [
        "f", "desc", "photo", "preview", "dropText", "drop",
        "train", "station", "date", "go", "clear", "chips",
        "status", "understood", "results"
    ]
    for dom_id in required_ids:
        assert dom_id in collector.elements_by_id, f"Missing required DOM element with id='{dom_id}'"

    # Verify Form attributes and inputs
    f_tag, f_attrs = collector.elements_by_id["f"]
    assert f_tag == "form", f"Element with id='f' should be <form>, found <{f_tag}>"

    desc_tag, desc_attrs = collector.elements_by_id["desc"]
    assert desc_tag == "textarea"
    assert desc_attrs.get("name") == "description", "Textarea #desc must have name='description'"
    assert desc_attrs.get("maxlength") == "500", "Textarea #desc must have maxlength='500'"

    photo_tag, photo_attrs = collector.elements_by_id["photo"]
    assert photo_tag == "input"
    assert photo_attrs.get("name") == "photo", "Input #photo must have name='photo'"
    assert photo_attrs.get("type") == "file", "Input #photo must have type='file'"

    train_tag, train_attrs = collector.elements_by_id["train"]
    assert train_attrs.get("name") == "train_no", "Element #train must have name='train_no'"

    station_tag, station_attrs = collector.elements_by_id["station"]
    assert station_attrs.get("name") == "station", "Element #station must have name='station'"

    date_tag, date_attrs = collector.elements_by_id["date"]
    assert date_tag == "input"
    assert date_attrs.get("name") == "lost_date", "Element #date must have name='lost_date'"
    assert date_attrs.get("type") == "date", "Element #date must have type='date'"

    go_tag, go_attrs = collector.elements_by_id["go"]
    assert go_attrs.get("type") == "submit", "Button #go must have type='submit'"

    clear_tag, clear_attrs = collector.elements_by_id["clear"]
    assert clear_attrs.get("type") == "reset", "Button #clear must have type='reset'"


def test_accessibility_and_responsive_css(client):
    """Verify accessibility tokens, meter attributes, and responsive layout."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text

    # Viewport meta tag
    assert 'name="viewport"' in text
    assert "width=device-width" in text

    # Focus-visible 3px rust outline
    assert ":focus-visible" in text
    assert "3px solid" in text
    assert "#9A4320" in text.upper()

    # Match meter accessibility attributes in template
    assert 'role="meter"' in text
    assert 'aria-label="Match strength"' in text
    assert 'aria-valuemin="0"' in text
    assert 'aria-valuemax="100"' in text
    assert 'aria-valuenow' in text

    # Responsive Media Query
    assert "@media (max-width: 768px)" in text or "@media (max-width:" in text

    # Filter toggle ARIA attributes
    assert 'aria-expanded="false"' in text
    assert 'aria-controls="filters"' in text


def test_meta_api_integration(client):
    """Verify GET /api/meta provides dynamic metadata for frontend dropdowns."""
    r = client.get("/api/meta")
    assert r.status_code == 200
    meta = r.json()

    assert "stations" in meta and isinstance(meta["stations"], list)
    assert len(meta["stations"]) > 0
    assert "trains" in meta and isinstance(meta["trains"], list)
    assert len(meta["trains"]) > 0
    assert "train_no" in meta["trains"][0] and "train_name" in meta["trains"][0]
    assert "categories" in meta and len(meta["categories"]) > 0


def test_search_and_image_pipeline_integration(client):
    """Verify end-to-end POST /api/search response matches frontend rendering requirements."""
    r = client.post("/api/search", data={"description": "backpack with keychain", "top_k": 4})
    assert r.status_code == 200
    data = r.json()

    assert "query_id" in data and data["query_id"]
    assert "latency_ms" in data
    assert "understood" in data
    assert "results" in data and len(data["results"]) > 0

    first_item = data["results"][0]
    for required_key in ["item_id", "strength", "score", "category", "station", "train_no", "image_url"]:
        assert required_key in first_item, f"Result item missing field: {required_key}"

    # Verify that the image_url served can be successfully fetched
    img_resp = client.get(first_item["image_url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/jpeg"


def test_feedback_claim_and_undo_flow(client):
    """Verify POST /api/feedback handles both 'this_is_mine' and 'not_mine' actions."""
    # Obtain a genuine item_id from search
    search_resp = client.post("/api/search", data={"description": "umbrella"})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data["results"]) > 0

    target = search_data["results"][0]
    query_id = search_data["query_id"]
    item_id = target["item_id"]

    # Claim action: this_is_mine
    claim_resp = client.post("/api/feedback", json={
        "query_id": query_id,
        "item_id": item_id,
        "rank": 1,
        "action": "this_is_mine"
    })
    assert claim_resp.status_code == 200
    assert claim_resp.json() == {"ok": True}

    # Undo action: not_mine
    undo_resp = client.post("/api/feedback", json={
        "query_id": query_id,
        "item_id": item_id,
        "rank": 1,
        "action": "not_mine"
    })
    assert undo_resp.status_code == 200
    assert undo_resp.json() == {"ok": True}


def test_client_scripts_and_deep_linking_contracts(client):
    """Verify client-side script includes search, drag-and-drop, deep linking, and undo logic."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text

    # Deep linking via URL parameter
    assert "URLSearchParams(location.search).get" in text or "location.search" in text
    assert '"q"' in text or "'q'" in text

    # Drag and drop listeners
    assert "dragover" in text
    assert "dragleave" in text
    assert "drop" in text

    # Search without filters
    assert "btn-no-filters" in text or "Search without filters" in text

    # Claim feedback endpoints
    assert "/api/feedback" in text
    assert "this_is_mine" in text
    assert "not_mine" in text

    # Developer links in footer
    assert "/health" in text
    assert "/metrics" in text
    assert "/docs" in text


def test_search_with_train_station_date_filters(client):
    """Verify POST /api/search accurately parses secondary filter fields."""
    r = client.post("/api/search", data={
        "description": "Wildcraft backpack",
        "train_no": "12951",
        "station": "Borivali",
        "lost_date": "2026-11-01"
    })
    assert r.status_code == 200
    body = r.json()
    u = body["understood"]
    assert u["train_no"] == "12951"
    assert u["station"] == "Borivali"
    assert u["lost_date"] == "2026-11-01"
    assert "results" in body


def test_multimodal_search_with_image_and_text(client):
    """Verify POST /api/search supports combined image and text multimodal input."""
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (30, 140, 60)).save(buf, "JPEG")
    photo_bytes = buf.getvalue()

    r = client.post(
        "/api/search",
        data={"description": "green umbrella"},
        files={"photo": ("umbrella.jpg", photo_bytes, "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["understood"]["used_photo"] is True
    assert "results" in body


def test_search_bad_request_validations(client):
    """Verify server validation responses match expected contract for error states."""
    # Missing both description and photo
    r1 = client.post("/api/search", data={})
    assert r1.status_code == 400

    # Invalid date format
    r2 = client.post("/api/search", data={"description": "bag", "lost_date": "14-11-2026"})
    assert r2.status_code == 400


def test_semantic_landmarks_and_aside(client):
    """Verify semantic page hierarchy: header, main, how-it-works aside, and footer."""
    r = client.get("/")
    assert r.status_code == 200
    text = r.text

    assert '<header class="site-header">' in text or "<header" in text
    assert '<main id="top"' in text
    assert '<aside id="how"' in text
    assert '<footer id="data"' in text
    assert 'href="#how"' in text
    assert 'href="#data"' in text
