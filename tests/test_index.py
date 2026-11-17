from datetime import date

import numpy as np
import pytest

from khoya.index import GalleryIndex


def make_index():
    emb = np.eye(4, dtype=np.float32)
    items = [
        {"item_id": f"I{i}", "category": "bag", "station": st, "train_no": tr, "train_name": "",
         "found_date": fd}
        for i, (st, tr, fd) in enumerate([
            ("Dadar", "12951", "2026-11-01"),
            ("Surat", "12009", "2026-11-05"),
            ("Dadar", "12123", "2026-10-01"),
            ("Pune Junction", "12123", "2026-11-09"),
        ])
    ]
    return GalleryIndex(emb, items, {})


def test_best_match_first():
    hits = make_index().search(image_emb=np.array([0, 0.9, 0.1, 0], dtype=np.float32), top_k=2)
    assert [h.item["item_id"] for h in hits] == ["I1", "I2"]


def test_train_boost_reorders_close_scores():
    q = np.array([0.5, 0.49, 0, 0], dtype=np.float32)
    hits = make_index().search(text_emb=q, train_no="12009", top_k=1, train_boost=1.0)
    assert hits[0].item["item_id"] == "I1"
    assert "same train" in hits[0].reasons


def test_date_window_filters():
    q = np.array([0, 0, 1, 0], dtype=np.float32)  # best raw match was found BEFORE the loss
    hits = make_index().search(text_emb=q, lost_date=date(2026, 10, 30), max_days_after_loss=30)
    ids = {h.item["item_id"] for h in hits}
    assert "I2" not in ids and ids == {"I0", "I1", "I3"}


def test_needs_a_query():
    with pytest.raises(ValueError):
        make_index().search()


def test_roundtrip(tmp_path):
    idx = make_index()
    idx.save(tmp_path)
    again = GalleryIndex.load(tmp_path)
    assert again.items == idx.items and np.allclose(again.embeddings, idx.embeddings)
    assert again.stations == ["Dadar", "Pune Junction", "Surat"]
