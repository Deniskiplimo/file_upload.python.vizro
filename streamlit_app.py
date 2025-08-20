from __future__ import annotations
import os, io, sys
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from data_utils import (
    read_any,
    basic_profile,
    pick_columns_for_charts,
    summarize_trend,
    save_temp_dataset,
)

# ---------------- Config ----------------
st.set_page_config(page_title="📊 Upload → Visualize → Explain", layout="wide")
st.title("📊 Upload → Visualize → Explain (Streamlit + Vizro + API)")
st.caption(
    "Upload a dataset (CSV/XLSX/JSON/Parquet). We'll profile it, generate charts, "
    "write plain-English insights, and export for a Vizro dashboard."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Options")
    sample = st.toggle("Use Plotly sample (Iris)", value=False)
    enable_downsample = st.toggle("Downsample large datasets", value=True)
    downsample_rows = st.number_input(
        "Rows when downsampling", min_value=500, max_value=20000, value=5000, step=500
    )
    corr_method = st.radio("Correlation method", ["pearson", "spearman"], index=0)
    agg_func = st.selectbox("Aggregation for category bars", ["mean", "sum", "median"])

# ---------------- File Upload ----------------
uploaded = st.file_uploader(
    "Drop data file(s)", 
    type=["csv", "xlsx", "xls", "json", "parquet"], 
    accept_multiple_files=True
)

@st.cache_data
def load_data(file_bytes, filename):
    return read_any(file_bytes, filename)

if sample and not uploaded:
    import plotly.express as _px
    df = _px.data.iris()
    filename = "sample_iris.csv"
else:
    if not uploaded:
        st.info("Upload a file to begin, or toggle the sample.")
        st.stop()

    # Combine multiple files if provided
    dfs = []
    for file in uploaded:
        file_bytes = file.read()
        filename = file.name
        try:
            dfs.append(load_data(file_bytes, filename))
        except Exception as e:
            st.error(f"❌ Failed to read {filename}: {e}")
            st.stop()
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# ---------------- Downsampling ----------------
if enable_downsample and len(df) > downsample_rows:
    df = df.sample(downsample_rows, random_state=42).reset_index(drop=True)
    st.warning(f"Dataset downsampled to {downsample_rows} rows for responsiveness.")

st.success(f"Loaded **{filename}** → {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------- Preview ----------------
with st.expander("🔍 Preview (first 100 rows)", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

# ---------------- Profile ----------------
prof = basic_profile(df)
with st.expander("🧭 Column profile", expanded=False):
    st.json({"shape": prof["shape"]})
    st.dataframe(pd.DataFrame(prof["columns"]), use_container_width=True)

    # Missing values
    missing = df.isna().mean().sort_values(ascending=False)
    if missing.any():
        st.subheader("Missing values (%)")
        st.bar_chart(missing)

# ---------------- Filters ----------------
st.subheader("🔎 Quick Filters")
cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique(dropna=True) <= 50]
num_cols = df.select_dtypes(include=np.number).columns.tolist()
date_cols = [c for c in df.columns if "date" in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)]

filters = {}
if cat_cols or num_cols or date_cols:
    with st.expander("Add filters"):
        if cat_cols:
            st.markdown("**Categorical filters**")
            for c in cat_cols[:6]:
                pick = st.multiselect(f"{c}", sorted(df[c].dropna().unique()), default=[])
                if pick:
                    filters[c] = pick
        if num_cols:
            st.markdown("**Numeric filters**")
            for c in num_cols[:4]:
                min_val, max_val = float(df[c].min()), float(df[c].max())
                rng = st.slider(f"{c}", min_val, max_val, (min_val, max_val))
                df = df[df[c].between(rng[0], rng[1])]
        if date_cols:
            st.markdown("**Date filters**")
            for c in date_cols:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    dmin, dmax = df[c].min(), df[c].max()
                    dr = st.date_input(f"{c} range", [dmin, dmax])
                    if len(dr) == 2:
                        df = df[(df[c] >= dr[0]) & (df[c] <= dr[1])]
                except Exception:
                    pass

for c, vals in filters.items():
    df = df[df[c].isin(vals)]

# ---------------- Suggested charts ----------------
st.subheader("📈 Auto-generated visuals")
sugs = pick_columns_for_charts(df)

# Time series
if sugs["time_series"]:
    st.markdown("**Time series**")
    for spec in sugs["time_series"]:
        t, y = spec["x"], spec["y"]
        try:
            fig = px.line(df.sort_values(t), x=t, y=y, title=f"{y} over {t}")
            st.plotly_chart(fig, use_container_width=True)
            expl = summarize_trend(df, t, y)
            if expl:
                st.markdown("🧠 " + expl)
        except Exception as e:
            st.info(f"Skipped time series {y} vs {t}: {e}")

# Bars
if sugs["bars"]:
    st.markdown("**Category bars**")
    for spec in sugs["bars"]:
        c, y = spec["x"], spec["y"]
        try:
            g = df.groupby(c)[y].agg(agg_func).sort_values(ascending=False).reset_index()
            fig = px.bar(g, x=c, y=y, title=f"{agg_func} {y} by {c}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Skipped bar {y} by {c}: {e}")

# Boxplots
if sugs["bars"]:
    st.markdown("**Boxplots**")
    for spec in sugs["bars"][:4]:
        c, y = spec["x"], spec["y"]
        try:
            fig = px.box(df, x=c, y=y, points="all", title=f"{y} distribution by {c}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# Histograms
if sugs["hists"]:
    st.markdown("**Distributions**")
    cols = st.columns(2)
    for i, spec in enumerate(sugs["hists"]):
        x = spec["x"]
        with cols[i % 2]:
            try:
                fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# Scatter plots
if sugs["scatters"]:
    st.markdown("**Relationships**")
    for spec in sugs["scatters"]:
        x, y = spec["x"], spec["y"]
        try:
            fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# Correlation heatmap
if sugs["heatmap"]:
    st.markdown("**Correlation heatmap**")
    cols = sugs["heatmap"]["cols"]
    try:
        corr = df[cols].corr(method=corr_method, numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"Correlation ({corr_method})")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Skipped heatmap: {e}")

# ---------------- Export ----------------
st.subheader("🚚 Export data")
vizro_path = save_temp_dataset(df)
st.code(f"Saved to: {vizro_path}")

st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name="export.csv")
to_excel = io.BytesIO()
df.to_excel(to_excel, index=False, engine="xlsxwriter")
st.download_button("⬇️ Download XLSX", data=to_excel.getvalue(), file_name="export.xlsx")

st.markdown(
    "Launch Vizro on this data in a terminal:\n\n"
    f"```bash\npython vizro_app.py --data \"{vizro_path}\" --port 8051\n```"
)
