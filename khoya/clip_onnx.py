"""CLIP inference with ONNX Runtime only — no PyTorch at serving time.

CLIP maps images and text into the same embedding space, so
"black backpack with red keychain" lands near photos of such a backpack.
We L2-normalise every embedding, so a dot product is a cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]
MAX_TOKENS = 77


def model_files(variant: str) -> dict[str, str]:
    """Files we need from the Hugging Face repo, for a given precision variant."""
    suffix = "" if variant == "fp32" else f"_{variant}"
    return {
        "vision": f"onnx/vision_model{suffix}.onnx",
        "text": f"onnx/text_model{suffix}.onnx",
        "tokenizer": "tokenizer.json",
        "preprocessor": "preprocessor_config.json",
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.clip(norm, 1e-12, None)).astype(np.float32)


def _pick_output(session: ort.InferenceSession, preferred: str) -> str:
    outputs = session.get_outputs()
    names = [o.name for o in outputs]
    if preferred in names:
        return preferred
    for o in outputs:  # fall back to the first 2-D output: (batch, dim)
        if len(o.shape) == 2:
            return o.name
    return names[0]


class ClipEncoder:
    def __init__(self, model_dir: str | Path, variant: str = "quantized", threads: int = 1):
        model_dir = Path(model_dir)
        files = model_files(variant)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(str(model_dir / files["vision"]), opts, providers=providers)
        self.text = ort.InferenceSession(str(model_dir / files["text"]), opts, providers=providers)

        self.tokenizer = Tokenizer.from_file(str(model_dir / files["tokenizer"]))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)

        cfg = {}
        pp = model_dir / files["preprocessor"]
        if pp.exists():
            cfg = json.loads(pp.read_text(encoding="utf-8"))
        size = cfg.get("size", 224)
        if isinstance(size, dict):
            size = size.get("shortest_edge", size.get("height", 224))
        crop = cfg.get("crop_size", size)
        if isinstance(crop, dict):
            crop = crop.get("height", size)
        self.resize_to, self.crop = int(size), int(crop)
        self.mean = np.array(cfg.get("image_mean", DEFAULT_MEAN), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg.get("image_std", DEFAULT_STD), dtype=np.float32).reshape(3, 1, 1)

        vis_input = self.vision.get_inputs()[0]
        self._vis_in = vis_input.name
        batch_dim = vis_input.shape[0]
        self._vis_batch = 1 if batch_dim == 1 else 16  # some exports fix batch size to 1
        self._vis_out = _pick_output(self.vision, "image_embeds")
        self._txt_inputs = {i.name for i in self.text.get_inputs()}
        self._txt_out = _pick_output(self.text, "text_embeds")

    # ---------- images ----------
    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Same steps as Hugging Face's CLIPImageProcessor: resize shortest side,
        centre-crop, scale to [0,1], normalise with CLIP's mean/std, HWC -> CHW."""
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        scale = self.resize_to / min(w, h)
        new_w, new_h = max(self.crop, round(w * scale)), max(self.crop, round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        left, top = (new_w - self.crop) // 2, (new_h - self.crop) // 2
        img = img.crop((left, top, left + self.crop, top + self.crop))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (arr - self.mean) / self.std

    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        pixels = [self.preprocess(im) for im in images]
        out = []
        for i in range(0, len(pixels), self._vis_batch):
            batch = np.stack(pixels[i : i + self._vis_batch]).astype(np.float32)
            (emb,) = self.vision.run([self._vis_out], {self._vis_in: batch})
            out.append(emb)
        return l2_normalize(np.concatenate(out, axis=0))

    def encode_image(self, img: Image.Image) -> np.ndarray:
        return self.encode_images([img])[0]

    # ---------- text ----------
    def encode_text(self, text: str) -> np.ndarray:
        # One query at a time = no padding, which keeps CLIP's end-of-text pooling exact.
        ids = np.array([self.tokenizer.encode(text).ids], dtype=np.int64)
        feeds = {}
        for name in self._txt_inputs:
            if name == "input_ids":
                feeds[name] = ids
            elif name == "attention_mask":
                feeds[name] = np.ones_like(ids)
            elif name == "position_ids":
                feeds[name] = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        (emb,) = self.text.run([self._txt_out], feeds)
        return l2_normalize(emb)[0]
