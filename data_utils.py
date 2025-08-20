from __future__ import annotations
import io, os, json, uuid
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".feather"}


# ---------------- File Readers ----------------
def read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read any supported file type into a DataFrame from bytes."""
    name = filename.lower()
    bio = io.BytesIO(file_bytes)

    try:
        if name.endswith(".csv"):
            return pd.read_csv(bio)
        if name.endswith(".tsv"):
            return pd.read_csv(bio, sep="\t")
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(bio)
        if name.endswith(".json"):
            obj = json.load(bio)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            return pd.json_normalize(obj, sep=".")
        if name.endswith(".parquet"):
            return pd.read_parquet(bio)
        if name.endswith(".feather"):
            return pd.read_feather(bio)
    except Exception as e:
        raise ValueError(f"❌ Failed to read {filename}: {e}") from e

    raise ValueError(f"Unsupported file type for {filename}")


def read_path(path: str) -> pd.DataFrame:
    """Read a dataset from disk path."""
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if p.endswith(".json"):
        return pd.read_json(path)
    if p.endswith(".feather"):
        return pd.read_feather(path)
    raise ValueError("Unsupported file type")


# ---------------- Role Inference ----------------
def infer_role(s: pd.Series) -> str:
    """Infer semantic role of a column: time, numeric, category, text, etc."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        unique_ratio = s.nunique(dropna=True) / max(len(s), 1)
        if unique_ratio < 0.02 or s.nunique() < 20:
            return "category_numeric"
        return "numeric"
    if s.dtype == object:
        # try parse dates
        try:
            pd.to_datetime(s.dropna().sample(min(50, len(s))), errors="raise")
            return "time"
        except Exception:
            pass
        nunique = s.nunique(dropna=True)
        if nunique <= 50:
            return "category"
        if nunique > len(s) * 0.9:
            return "id"  # mostly unique identifiers
        return "text"
    return "other"


# ---------------- Profiling ----------------
def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic dataset profile with enriched column stats."""
    cols = []
    for c in df.columns:
        s = df[c]
        role = infer_role(s)
        col_info = {
            "name": c,
            "dtype": str(s.dtype),
            "role": role,
            "nulls": int(s.isna().sum()),
            "null_pct": float(s.isna().mean()),
            "distinct": int(s.nunique(dropna=True)),
            "sample": s.dropna().astype(str).head(3).tolist(),
        }
        if role in {"numeric", "category_numeric"}:
            desc = s.describe()
            col_info.update({
                "min": float(desc.get("min", np.nan)),
                "max": float(desc.get("max", np.nan)),
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
            })
        if role in {"category", "text"}:
            top_vals = s.value_counts().head(3).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        cols.append(col_info)

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": cols,
    }


# ---------------- Chart Suggestions ----------------
def pick_columns_for_charts(df: pd.DataFrame) -> Dict[str, Any]:
    """Suggest useful chart specs based on inferred column roles."""
    roles = {c: infer_role(df[c]) for c in df.columns}
    time_cols = [c for c, r in roles.items() if r == "time"]
    num_cols = [c for c, r in roles.items() if r in ("numeric", "category_numeric")]
    cat_cols = [c for c, r in roles.items() if r == "category"]

    suggestions = {"time_series": [], "bars": [], "scatters": [], "hists": [], "boxplots": [], "heatmap": None}

    # time series
    if time_cols:
        t = time_cols[0]
        for y in num_cols[:6]:
            suggestions["time_series"].append({"x": t, "y": y, "kind": "line"})

    # bar charts (agg)
    if cat_cols and num_cols:
        for y in num_cols[:2]:
            for c in cat_cols[:3]:
                suggestions["bars"].append({"x": c, "y": y, "agg": "mean", "kind": "bar"})

    # boxplots
    if cat_cols and num_cols:
        for y in num_cols[:2]:
            for c in cat_cols[:2]:
                suggestions["boxplots"].append({"x": c, "y": y, "kind": "box"})

    # pairwise scatter (prioritize correlated pairs)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs().unstack().sort_values(ascending=False)
        seen = set()
        for (a, b), val in corr.items():
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            suggestions["scatters"].append({"x": a, "y": b, "kind": "scatter", "corr": float(val)})
            if len(suggestions["scatters"]) >= 6:
                break

    # histograms
    for c in num_cols[:8]:
        suggestions["hists"].append({"x": c, "kind": "hist"})

    # heatmap
    if len(num_cols) >= 3:
        suggestions["heatmap"] = {"cols": num_cols[:12], "kind": "corr"}

    return suggestions


# ---------------- Trend Summary ----------------
def summarize_trend(df: pd.DataFrame, time_col: Optional[str], y_col: Optional[str]) -> Optional[str]:
    """Simple linear trend summary with direction and variability."""
    if time_col is None or y_col is None:
        return None
    try:
        ts = df[[time_col, y_col]].dropna().copy()
        ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
        ts = ts.dropna().sort_values(time_col)
        if len(ts) < 5:
            return None

        y = ts[y_col].values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
        volatility = np.std(y) / (np.mean(y) + 1e-9)
        return f"Over time, **{y_col}** appears {direction} (linear trend, volatility={volatility:.2f})."
    except Exception:
        return None


# ---------------- Save Temp ----------------
def save_temp_dataset(df: pd.DataFrame, base_dir: str = ".vizro_cache") -> str:
    """Save dataset to both Parquet and CSV with unique filenames, return Parquet path."""
    os.makedirs(base_dir, exist_ok=True)
    uid = str(uuid.uuid4())[:8]
    path_parquet = os.path.join(base_dir, f"uploaded_{uid}.parquet")
    path_csv = os.path.join(base_dir, f"uploaded_{uid}.csv")
    df.to_parquet(path_parquet, index=False)
    df.to_csv(path_csv, index=False)
    return os.path.abspath(path_parquet)
