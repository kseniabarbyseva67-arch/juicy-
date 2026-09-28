from __future__ import annotations
import numpy as np
from .storage import GalleryStore, StoredItem


def cosine_search(store: GalleryStore, query: np.ndarray, top_k: int) -> list[tuple[StoredItem, float]]:
    items = store.all()
    if not items:
        return []
    mat = np.stack([i.embedding for i in items])
    scores = mat @ query  # both sides are L2-normalized
    order = np.argsort(-scores)[:top_k]
    return [(items[int(i)], float(scores[int(i)])) for i in order]
