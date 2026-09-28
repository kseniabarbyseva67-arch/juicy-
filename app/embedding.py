"""Offline vehicle embedding baseline.

The interface is intentionally isolated so a learned ReID model can replace the
baseline without changing API/storage code. This implementation uses only visual
appearance inside the supplied bounding box and never reads licence-plate text.
"""
from __future__ import annotations

import cv2
import numpy as np
from .config import EMBEDDING_DIM


class VehicleEmbedder:
    dim = EMBEDDING_DIM

    def _crop(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        ih, iw = image.shape[:2]
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x >= iw or y >= ih or x2 <= x or y2 <= y:
            raise ValueError("BBox is outside image bounds")
        return image[y:y2, x:x2]

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return v if norm == 0 else v / norm

    def embed(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        crop = self._crop(image, bbox)
        crop = cv2.resize(crop, (192, 128), interpolation=cv2.INTER_AREA)

        # 1) HSV colour histograms: robust-ish to moderate illumination changes.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
        hist = self._l2(hist.astype(np.float32))

        # 2) Coarse spatial colour moments preserve rough layout / vehicle paint.
        small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
        spatial = (small.astype(np.float32) / 255.0).reshape(-1)

        # 3) Silhouette / geometry via edge projections.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180).astype(np.float32) / 255.0
        row_proj = cv2.resize(edges.mean(axis=1)[:, None], (1, 32), interpolation=cv2.INTER_AREA).flatten()
        col_proj = cv2.resize(edges.mean(axis=0)[None, :], (32, 1), interpolation=cv2.INTER_AREA).flatten()

        # Concatenate, then deterministically project/pad to fixed dim.
        raw = np.concatenate([hist, spatial, row_proj, col_proj]).astype(np.float32)
        if raw.size >= self.dim:
            # Average pooling preserves information better than truncation.
            chunks = np.array_split(raw, self.dim)
            vec = np.array([c.mean() for c in chunks], dtype=np.float32)
        else:
            vec = np.pad(raw, (0, self.dim - raw.size)).astype(np.float32)
        return self._l2(vec)
