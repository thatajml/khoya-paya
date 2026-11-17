import csv
import json

from PIL import Image


def test_ingest_outputs(built):
    rows = list(csv.DictReader(open(built / "data/found_items.csv", encoding="utf-8")))
    assert len(rows) == 12
    assert {r["category"] for r in rows} == {"backpack", "umbrella", "suitcase", "wallet"}
    assert all(r["item_id"].startswith("FI-") and r["train_no"] and r["found_date"] for r in rows)
    img = Image.open(built / "data/gallery" / rows[0]["image"])
    assert max(img.size) <= 512
    assert len(img.getexif()) == 0, "EXIF (incl. GPS) must be stripped"


def test_ingest_is_stable_and_keeps_hand_edits(built):
    from pipeline import ingest

    path = built / "data/found_items.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rows[0]["officer_note"] = "Edited by hand"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ingest.FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert ingest.main() == 0
    again = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [r["item_id"] for r in again] == [r["item_id"] for r in rows]
    assert again[0]["officer_note"] == "Edited by hand"


def test_index_and_metrics(built):
    manifest = json.loads((built / "artifacts/index/manifest.json").read_text())
    assert manifest["n_items"] == 12 and manifest["dim"] == 16
    metrics = json.loads((built / "artifacts/metrics.json").read_text())["metrics"]
    for key in ("img2img_recall_at_1", "img2img_recall_at_5", "img2img_mrr",
                "text_category_precision_at_5", "latency_image_query_p50_ms"):
        assert key in metrics
    assert 0 <= metrics["img2img_recall_at_5"] <= 1
