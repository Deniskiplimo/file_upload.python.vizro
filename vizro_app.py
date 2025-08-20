from __future__ import annotations
import argparse
import pandas as pd
import vizro.models as vm
from vizro import Vizro
import vizro.plotly.express as px  # Vizro-wrapped Plotly Express

from data_utils import infer_role


# ---------------- Helpers ----------------
def make_overview(df: pd.DataFrame) -> vm.Page:
    return vm.Page(
        title="Overview",
        components=[
            vm.Text(text=f"### Dataset Overview\n**Rows:** {len(df):,} | **Columns:** {df.shape[1]}"),
            vm.Table(data=df.head(20)),
        ],
    )


def make_distributions(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> vm.Page:
    comps = []
    for col in num_cols[:10]:
        comps.append(vm.Graph(figure=px.histogram(df, x=col), title=f"Distribution of {col}"))

    for col in cat_cols[:3]:
        vc = df[col].value_counts().nlargest(15).reset_index()
        vc.columns = [col, "count"]
        comps.append(vm.Graph(figure=px.bar(vc, x=col, y="count"), title=f"Top categories in {col}"))

    return vm.Page(title="Distributions", components=comps, controls=[vm.Filter(column=c) for c in cat_cols[:1]])


def make_relationships(df: pd.DataFrame, num_cols: list[str]) -> vm.Page:
    comps = []
    for i in range(min(len(num_cols), 5)):
        for j in range(i + 1, min(len(num_cols), 5)):
            x, y = num_cols[i], num_cols[j]
            comps.append(vm.Graph(figure=px.scatter(df, x=x, y=y, trendline="ols"), title=f"{y} vs {x}"))
            if len(comps) >= 8:
                break
        if len(comps) >= 8:
            break
    return vm.Page(title="Relationships", components=comps)


def make_time_series(df: pd.DataFrame, time_cols: list[str], num_cols: list[str]) -> vm.Page:
    comps = []
    if time_cols:
        t = time_cols[0]
        dft = df.copy()
        dft[t] = pd.to_datetime(dft[t], errors="coerce")
        dft = dft.dropna(subset=[t])
        for y in num_cols[:6]:
            comps.append(vm.Graph(figure=px.line(dft.sort_values(t), x=t, y=y), title=f"{y} over {t}"))
    return vm.Page(title="Time Series", components=comps, controls=[vm.Filter(column=time_cols[0])] if time_cols else [])


def make_heatmap(df: pd.DataFrame, num_cols: list[str]) -> vm.Page:
    comps = []
    if len(num_cols) >= 3:
        corr = df[num_cols[:12]].corr(numeric_only=True, method="pearson")
        comps.append(vm.Graph(figure=px.imshow(corr, text_auto=True, aspect="auto"), title="Correlation heatmap"))
        comps.append(vm.Table(data=corr.round(3).reset_index(), title="Correlation table"))
    return vm.Page(title="Correlation", components=comps)


# ---------------- Dashboard Builder ----------------
def build_dashboard(df: pd.DataFrame) -> vm.Dashboard:
    roles = {c: infer_role(df[c]) for c in df.columns}
    time_cols = [c for c, r in roles.items() if r == "time"]
    num_cols = [c for c, r in roles.items() if r in ("numeric", "category_numeric")]
    cat_cols = [c for c, r in roles.items() if r == "category"]

    pages = [
        make_overview(df),
        make_distributions(df, num_cols, cat_cols),
        make_relationships(df, num_cols),
        make_time_series(df, time_cols, num_cols),
        make_heatmap(df, num_cols),
    ]

    return vm.Dashboard(
        pages=[p for p in pages if p.components],  # skip empty pages
        title="📊 Vizro: Upload → Multi-Page Dashboard",
    )


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet/JSON/XLSX")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()

    # Read by extension
    p = args.data.lower()
    if p.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    elif p.endswith(".csv"):
        df = pd.read_csv(args.data)
    elif p.endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.data)
    elif p.endswith(".json"):
        df = pd.read_json(args.data)
    else:
        raise SystemExit("❌ Unsupported format. Use CSV/XLSX/JSON/Parquet.")

    dashboard = build_dashboard(df)
    app = Vizro().build(dashboard)
    app.run(port=args.port)


if __name__ == "__main__":
    main()
