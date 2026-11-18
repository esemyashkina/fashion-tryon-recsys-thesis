import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel


class FashionCLIPEncoder:
    def __init__(
        self,
        model_id: str = "patrickjohncyh/fashion-clip",
        device: str | None = None,
        dtype: str = "float16",
        use_fast_processor: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.torch_dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=use_fast_processor)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.inference_mode()
    def encode_image_pil(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.torch_dtype)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        feats = self.model.visual_projection(pooled)

        feats = self._l2_normalize(feats).squeeze(0)
        return feats.float().cpu().numpy()


    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_outputs.pooler_output
        feats = self.model.text_projection(pooled)

        feats = self._l2_normalize(feats).squeeze(0)
        return feats.float().cpu().numpy()

    @torch.inference_mode()
    def encode_image_path(self, path: str) -> np.ndarray:
        with Image.open(path) as image:
            return self.encode_image_pil(image)
    