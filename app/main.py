from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import DEFAULT_TOP_K, REFUSAL_THRESHOLD, MAX_IMAGE_MB, EMBEDDING_DIM
from .embedding import VehicleEmbedder
from .schemas import BBox, Candidate, GalleryItemResponse, GalleryStats, QueryResponse
from .search import cosine_search
from .storage import GalleryStore

app = FastAPI(
    title="Falcon Vehicle ReID Service",
    version="1.0.0",
    description="Open-set vehicle image retrieval without using licence-plate recognition.",
)

BASE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))
store = GalleryStore()
embedder = VehicleEmbedder()


def decode_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_MB} MB")
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Unsupported or corrupted image")
    return image


def parse_bbox(text: str, image: np.ndarray) -> tuple[int,int,int,int]:
    try:
        obj = json.loads(text)
        bbox = BBox(**obj)
    except Exception as e:
        raise HTTPException(422, f"Invalid bbox JSON: {e}")
    ih, iw = image.shape[:2]
    if bbox.x >= iw or bbox.y >= ih:
        raise HTTPException(422, "BBox starts outside image")
    return bbox.x, bbox.y, min(bbox.w, iw-bbox.x), min(bbox.h, ih-bbox.y)


@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "threshold": REFUSAL_THRESHOLD})


@app.get("/health")
def health():
    return {"status": "ok", "gallery_items": store.count(), "embedding_dim": EMBEDDING_DIM}


@app.get("/api/v1/gallery/stats", response_model=GalleryStats)
def gallery_stats():
    return GalleryStats(items=store.count(), embedding_dim=EMBEDDING_DIM)


@app.delete("/api/v1/gallery")
def clear_gallery():
    store.clear()
    return {"status": "cleared"}


@app.post("/api/v1/gallery/items", response_model=GalleryItemResponse)
async def add_gallery_item(
    image: UploadFile = File(...),
    bbox: str = Form(..., description='JSON: {"x":0,"y":0,"w":100,"h":100}'),
    image_id: str = Form(...),
    vehicle_id: str | None = Form(None),
):
    data = await image.read()
    img = decode_image(data)
    bb = parse_bbox(bbox, img)
    try:
        vec = embedder.embed(img, bb)
    except ValueError as e:
        raise HTTPException(422, str(e))
    item = store.add(image_id=image_id, vehicle_id=vehicle_id, bbox=bb, embedding=vec)
    return GalleryItemResponse(
        item_id=item.item_id,
        image_id=item.image_id,
        vehicle_id=item.vehicle_id,
        bbox=BBox(x=item.x,y=item.y,w=item.w,h=item.h),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(
    image: UploadFile = File(...),
    bbox: str = Form(...),
    top_k: int = Form(DEFAULT_TOP_K, ge=1, le=100),
    threshold: float = Form(REFUSAL_THRESHOLD, ge=-1.0, le=1.0),
):
    data = await image.read()
    img = decode_image(data)
    bb = parse_bbox(bbox, img)
    try:
        q = embedder.embed(img, bb)
    except ValueError as e:
        raise HTTPException(422, str(e))

    ranked = cosine_search(store, q, top_k)
    top_score = ranked[0][1] if ranked else None
    refused = top_score is None or top_score < threshold
    candidates = [] if refused else [
        Candidate(
            item_id=i.item_id,
            image_id=i.image_id,
            vehicle_id=i.vehicle_id,
            score=round(score, 6),
            bbox=BBox(x=i.x,y=i.y,w=i.w,h=i.h),
        )
        for i, score in ranked
    ]
    return QueryResponse(refused=refused, threshold=threshold, top_score=top_score, candidates=candidates)
