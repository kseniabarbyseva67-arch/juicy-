from pydantic import BaseModel, Field
from typing import List, Optional


class BBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class Candidate(BaseModel):
    item_id: str
    image_id: str
    vehicle_id: Optional[str] = None
    score: float
    bbox: BBox


class QueryResponse(BaseModel):
    refused: bool
    threshold: float
    top_score: Optional[float] = None
    candidates: List[Candidate]


class GalleryItemResponse(BaseModel):
    item_id: str
    image_id: str
    vehicle_id: Optional[str] = None
    bbox: BBox


class GalleryStats(BaseModel):
    items: int
    embedding_dim: int
