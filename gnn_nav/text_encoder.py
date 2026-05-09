import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = text.replace(".", "")
    return text


def deterministic_random_embedding(text: str, dim: int) -> torch.Tensor:
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    gen = torch.Generator()
    gen.manual_seed(seed)
    emb = torch.randn(dim, generator=gen, dtype=torch.float32)
    emb = emb / (emb.norm() + 1e-6)
    return emb


class TextEmbeddingCache:
    def __init__(
        self,
        cache_path: str = "data/gnn/text_embeddings.pt",
        dim: int = 384,
        device: str = "cpu",
        backend: str = "auto",
    ):
        self.cache_path = cache_path
        self.dim = int(dim)
        self.device = device
        self.backend = backend
        self.cache: Dict[str, torch.Tensor] = {}
        self.model = None
        self._load_cache()
        self._init_backend()

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            payload = torch.load(self.cache_path, map_location="cpu")
            if isinstance(payload, dict) and "cache" in payload:
                payload = payload["cache"]
            if isinstance(payload, dict):
                for key, value in payload.items():
                    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
                    if tensor.numel() == self.dim:
                        self.cache[str(key)] = tensor.cpu()
        except Exception:
            self.cache = {}

    def _init_backend(self):
        if self.backend not in ["auto", "sentence_transformers"]:
            return
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.environ.get("GNN_NAV_SENTENCE_MODEL", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception:
            self.model = None

    def _encode_with_backend(self, text: str) -> Optional[torch.Tensor]:
        if self.model is None:
            return None
        try:
            emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            emb = torch.as_tensor(emb, dtype=torch.float32).flatten()
            if emb.numel() == self.dim:
                return emb
            if emb.numel() > self.dim:
                emb = emb[: self.dim]
            else:
                emb = torch.cat([emb, torch.zeros(self.dim - emb.numel(), dtype=torch.float32)])
            return emb / (emb.norm() + 1e-6)
        except Exception:
            return None

    def encode(self, text: str) -> torch.Tensor:
        key = normalize_text(text)
        if key in self.cache:
            return self.cache[key].clone()

        emb = self._encode_with_backend(key)
        if emb is None:
            emb = deterministic_random_embedding(key, self.dim)
        emb = emb.detach().cpu().float()
        emb = emb / (emb.norm() + 1e-6)
        self.cache[key] = emb
        return emb.clone()

    def encode_many(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros((0, self.dim), dtype=torch.float32)
        return torch.stack([self.encode(text) for text in texts], dim=0)

    def save(self):
        if not self.cache_path:
            return
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(path) + ".tmp"
        torch.save({"dim": self.dim, "cache": self.cache}, tmp_path)
        os.replace(tmp_path, path)


if __name__ == "__main__":
    encoder = TextEmbeddingCache(cache_path="", dim=16, backend="fallback")
    a = encoder.encode("chair")
    b = encoder.encode("chair")
    print("same_text_equal", bool(torch.allclose(a, b)))
    print("shape", tuple(encoder.encode_many(["chair", "table"]).shape))
