from __future__ import annotations
import os, uuid, io, json, logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd

from data_utils import (
    read_any, read_path,
    basic_profile, pick_columns_for_charts,
    summarize_trend
)

# -------------------
# CONFIG
# -------------------
app = FastAPI(title="📊 Upload → Insights API")
DATA_DIR = ".api_store"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Enable CORS for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# MODELS
# -------------------
class UploadResponse(BaseModel):
    dataset_id: str
    path: str

class TrendRequest(BaseModel):
    dataset: str
    time_col: str
    y_col: str

# -------------------
# ROUTES
# -------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = read_any(content, file.filename)
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(400, f"Cannot read file: {e}")
    ds_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{ds_id}.parquet")
    df.to_parquet(path, index=False)
    logging.info(f"Dataset uploaded: {ds_id} ({len(df)} rows)")
    return UploadResponse(dataset_id=ds_id, path=os.path.abspath(path))

@app.get("/datasets")
def list_datasets() -> List[str]:
    return [f[:-8] for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    path = os.path.join(DATA_DIR, f"{dataset_id}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")
    os.remove(path)
    return {"deleted": dataset_id}

@app.get("/profile")
def profile(dataset: str = Query(...)):
    path = os.path.join(DATA_DIR, f"{dataset}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")
    df = read_path(path)
    return basic_profile(df)

@app.get("/suggestions")
def suggestions(dataset: str = Query(...)):
    path = os.path.join(DATA_DIR, f"{dataset}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")
    df = read_path(path)
    return pick_columns_for_charts(df)

@app.get("/sample")
def sample(dataset: str = Query(...), n: int = 20):
    path = os.path.join(DATA_DIR, f"{dataset}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")
    df = read_path(path)
    return {"rows": df.head(n).to_dict(orient="records")}

@app.post("/trend")
def trend(req: TrendRequest):
    path = os.path.join(DATA_DIR, f"{req.dataset}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")
    df = read_path(path)
    summary = summarize_trend(df, req.time_col, req.y_col)
    if not summary:
        raise HTTPException(400, "Could not compute trend")
    return {"trend": summary}

# -------------------
# DOWNLOADS
# -------------------
@app.get("/download/{dataset_id}")
def download(dataset_id: str, fmt: str = Query("csv", regex="^(csv|xlsx|parquet)$")):
    path = os.path.join(DATA_DIR, f"{dataset_id}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset not found")

    df = read_path(path)

    if fmt == "csv":
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_id}.csv"}
        )
    elif fmt == "xlsx":
        out_path = os.path.join(DATA_DIR, f"{dataset_id}.xlsx")
        df.to_excel(out_path, index=False)
        return FileResponse(out_path, filename=f"{dataset_id}.xlsx")
    elif fmt == "parquet":
        return FileResponse(path, filename=f"{dataset_id}.parquet")
