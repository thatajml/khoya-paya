"""Test fixtures: a complete mini-project in a temp folder.

Real CLIP is ~150 MB, so tests use tiny stand-in ONNX models with the same
input/output names. That lets CI test the pipeline and API in seconds, while
the real model's quality is checked separately by `pipeline.evaluate --gate`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper
from PIL import Image, ImageDraw
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from khoya.clip_onnx import model_files
from khoya.config import ROOT, load_params

DIM = 16
VOCAB = ["<unk>", "<bos>", "<eos>", "a", "photo", "of", "red", "blue", "green", "yellow",
         "backpack", "umbrella", "suitcase", "wallet", "black", "bag", "with", "keychain"]
CATEGORIES = ["backpack", "umbrella", "suitcase", "wallet"]


def _save(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def make_fake_vision(path: Path) -> None:
    # pixel_values -> 2x2 average pool (colour layout) -> 12 features -> linear -> image_embeds
    w = numpy_helper.from_array(np.random.default_rng(0).normal(size=(12, DIM)).astype(np.float32), "W")
    shape = numpy_helper.from_array(np.array([-1, 12], dtype=np.int64), "shape")
    nodes = [
        helper.make_node("AveragePool", ["pixel_values"], ["pooled"], kernel_shape=[112, 112], strides=[112, 112]),
        helper.make_node("Reshape", ["pooled", "shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "W"], ["image_embeds"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_vision",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["batch", DIM])],
        [w, shape],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def make_fake_text(path: Path) -> None:
    # input_ids -> embedding lookup -> mean over tokens -> text_embeds
    table = numpy_helper.from_array(np.random.default_rng(1).normal(size=(len(VOCAB), DIM)).astype(np.float32), "E")
    nodes = [
        helper.make_node("Gather", ["E", "input_ids"], ["tok"]),
        helper.make_node("ReduceMean", ["tok"], ["text_embeds"], axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "fake_text",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        [helper.make_tensor_value_info("text_embeds", TensorProto.FLOAT, ["batch", DIM])],
        [table],
    )
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def make_fake_tokenizer(path: Path) -> None:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>",
                                                       special_tokens=[("<bos>", 1), ("<eos>", 2)])
    tok.save(str(path))


def make_raw_photos(raw: Path) -> None:
    """3 photos per category, each a distinct coloured layout (+ an EXIF tag to check stripping)."""
    rng = random.Random(3)
    for cat in CATEGORIES:
        for i in range(3):
            img = Image.new("RGB", (400, 300), tuple(rng.randint(0, 255) for _ in range(3)))
            d = ImageDraw.Draw(img)
            for _ in range(3):
                x, y = rng.randint(0, 300), rng.randint(0, 200)
                d.rectangle([x, y, x + 120, y + 110], fill=tuple(rng.randint(0, 255) for _ in range(3)))
            out = raw / "own" / cat / f"{cat}_{i}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            exif = Image.Exif()
            exif[0x010F] = "TestPhoneMaker"
            img.save(out, "JPEG", exif=exif)


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("proj")
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    params["model"]["dir"] = "artifacts/model"
    params["data"]["human_eval_csv"] = "data/eval_queries.csv"
    params["eval"]["gates"] = {"img2img_recall_at_5": 0.3}

    files = model_files(params["model"]["variant"])
    model_dir = root / "artifacts/model"
    make_fake_vision(model_dir / files["vision"])
    make_fake_text(model_dir / files["text"])
    make_fake_tokenizer(model_dir / files["tokenizer"])
    make_raw_photos(root / "data/raw")
    (root / "params.yaml").write_text(yaml.safe_dump(params))
    return root


@pytest.fixture(scope="session")
def built(project, monkeypatch_session) -> Path:
    """Runs the real pipeline stages (ingest -> index -> evaluate) on the mini-project."""
    monkeypatch_session.setenv("KHOYA_PARAMS", str(project / "params.yaml"))
    load_params.cache_clear()
    from pipeline import build_index, evaluate, ingest

    assert ingest.main() == 0
    assert build_index.main() == 0
    import sys

    argv = sys.argv
    sys.argv = ["evaluate", "--gate"]
    try:
        assert evaluate.main() == 0
    finally:
        sys.argv = argv
    return project


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
    load_params.cache_clear()
