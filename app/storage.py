from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import DB_PATH, EMBEDDING_DIM


@dataclass
class StoredItem:
    item_id: str
    image_id: str
    vehicle_id: str | None
    x: int
    y: int
    w: int
    h: int
    embedding: np.ndarray


class GalleryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS gallery (
                    item_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    vehicle_id TEXT,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    w INTEGER NOT NULL,
                    h INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )"""
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_gallery_image_id ON gallery(image_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_gallery_vehicle_id ON gallery(vehicle_id)")

    def add(self, image_id: str, bbox: tuple[int,int,int,int], embedding: np.ndarray, vehicle_id: str | None = None) -> StoredItem:
        item_id = str(uuid.uuid4())
        x, y, w, h = bbox
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Expected embedding shape {(EMBEDDING_DIM,)}, got {vec.shape}")
        with self._connect() as con:
            con.execute(
                "INSERT INTO gallery VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, image_id, vehicle_id, x, y, w, h, vec.tobytes()),
            )
        return StoredItem(item_id, image_id, vehicle_id, x, y, w, h, vec)

    def all(self) -> list[StoredItem]:
        with self._connect() as con:
            rows = con.execute("SELECT item_id,image_id,vehicle_id,x,y,w,h,embedding FROM gallery").fetchall()
        out = []
        for r in rows:
            vec = np.frombuffer(r[7], dtype=np.float32).copy()
            out.append(StoredItem(*r[:7], vec))
        return out

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM gallery").fetchone()[0])

    def clear(self):
        with self._connect() as con:
            con.execute("DELETE FROM gallery")
