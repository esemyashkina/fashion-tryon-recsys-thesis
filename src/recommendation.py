"""Shared photo-query, FashionCLIP ranking and Terra selection."""

import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from .evaluate import ROOT

FASHIONCLIP_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"


def image_part(image):
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 768))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "low"}


def load_encoder():
    from huggingface_hub import snapshot_download
    from .encoder import FashionCLIPEncoder

    model = snapshot_download(
        "patrickjohncyh/fashion-clip",
        revision=FASHIONCLIP_REVISION,
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    encoder = FashionCLIPEncoder(model_id=model, device="cpu", dtype="float32", use_fast_processor=False)
    if sys.platform == "darwin":
        import torch

        with torch.no_grad():
            for parameter in encoder.model.parameters():
                parameter.data = parameter.detach().clone()
    encoder.processor.tokenizer.model_max_length = encoder.model.config.text_config.max_position_embeddings
    return encoder


def catalog_index(encoder, folder, catalog):
    """Open a persistent USearch index for this exact set of catalog images."""
    from usearch.index import Index

    skus = sorted(catalog)
    keys = {sku: key for key, sku in enumerate(skus)}
    images = [folder / catalog[sku]["image_path"] for sku in skus]
    fingerprint = sha256(FASHIONCLIP_REVISION.encode())
    for sku, image in zip(skus, images):
        fingerprint.update(sku.encode() + b"\0")
        fingerprint.update(sha256(image.read_bytes()).digest())

    directory = ROOT / "runs" / "indexes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{folder.name}-{fingerprint.hexdigest()}.usearch"
    if path.is_file():
        try:
            index = Index.restore(str(path))
            if index is not None and len(index) == len(keys):
                return index, keys
        except (OSError, RuntimeError, ValueError):
            pass

    vectors = [encoder.encode_image_path(image) for image in images]
    index = Index(ndim=len(vectors[0]), metric="cos", dtype="f32")
    for key, vector in enumerate(vectors):
        index.add(key, vector)
    with TemporaryDirectory(dir=directory) as temporary:
        staged = Path(temporary) / path.name
        index.save(str(staged))
        staged.replace(path)
    return index, keys


def rank(encoder, catalog_vectors, text, pool):
    index, keys = catalog_vectors
    vector = encoder.encode_text(text)
    scores = {sku: float(index.get(keys[sku]) @ vector) for sku in pool}
    return sorted(pool, key=lambda sku: (-scores[sku], sku))


class Terra:
    def __init__(self, model, reasoning_effort):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        self.client = OpenAI(timeout=90, max_retries=2)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.query_prompt = (ROOT / "prompts/query.txt").read_text(encoding="utf-8").strip()

    def ask(self, instructions, parts):
        response = self.client.responses.create(
            model=self.model, instructions=instructions, reasoning={"effort": self.reasoning_effort},
            max_output_tokens=2400, store=False, text={"format": {"type": "json_object"}},
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Return a JSON object."}, *parts]}],
        )
        return json.loads(response.output_text)

    def query(self, photo, filters):
        context = [
            {"type": "input_text", "text": f"Requested replacement: {json.dumps(filters)}. First image is the outfit."},
            image_part(photo),
        ]
        return self.ask(self.query_prompt + "\nReturn JSON with query and reason.", context), context

    def choose(self, prompt, context, shortlist, open_product):
        if len(shortlist) == 1:
            return {"sku": shortlist[0], "reason": "Only candidate matching the filters."}
        parts = list(context)
        for number, sku in enumerate(shortlist, 1):
            parts.append({"type": "input_text", "text": f"Product {number}"})
            parts.append(image_part(open_product(sku)))
        pick = self.ask(prompt, parts)
        number = pick["choice"]
        if type(number) is not int or not 1 <= number <= len(shortlist):
            raise ValueError("Terra returned an invalid candidate number.")
        return {"sku": shortlist[number - 1], "reason": pick["reason"]}
