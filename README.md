# Upload → Visualize → Explain (Streamlit) + Vizro + FastAPI

A complete data visualization stack:
- **Streamlit** app for upload → auto‑EDA → insights → export.
- **Vizro** (McKinsey) app for a polished multi‑page dashboard on the same data.
- **FastAPI** backend exposing upload & profiling endpoints for integration.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Run the API (optional, for programmatic access)
```bash
uvicorn api:app --reload --port 8000
```

### 2) Run Streamlit (upload + auto insights + one‑click Vizro command)
```bash
streamlit run streamlit_app.py
```

### 3) Run Vizro on the saved dataset
Copy the path Streamlit shows (e.g. `.vizro_cache/uploaded.parquet`) then:
```bash
python vizro_app.py --data "/absolute/path/to/uploaded.parquet" --port 8051
```

## API Endpoints

- `POST /upload` (multipart): upload CSV/XLSX/JSON/Parquet → `{dataset_id, path}`
- `GET /profile?dataset=ID` → basic schema/stats
- `GET /suggestions?dataset=ID` → chart suggestions/specs
- `GET /sample?dataset=ID&n=20` → preview rows
- `GET /health` → service status
# file_upload.python.vizro
