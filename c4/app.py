from flask import Flask, flash, redirect, render_template, request, url_for
import pandas as pd
from pathlib import Path
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext == ".xlsx":
        # pandas.read_excel needs an Excel engine (openpyxl for .xlsx).
        try:
            import openpyxl  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency for .xlsx files: install with `pip install openpyxl`."
            ) from e
        return pd.read_excel(filepath)
    elif ext == ".xls":
        # pandas.read_excel uses xlrd for legacy .xls.
        try:
            import xlrd  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency for .xls files: install with `pip install xlrd`."
            ) from e
        return pd.read_excel(filepath)
    elif ext == ".json":
        return pd.read_json(filepath)
    elif ext == ".xml":
        # pandas.read_xml requires lxml.
        try:
            import lxml  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency for XML files: install with `pip install lxml`."
            ) from e
        return pd.read_xml(filepath)
    else:
        raise ValueError("Unsupported file format")

def analyze_data(filepath):
    df = load_data(filepath)

    df.columns = [str(col).strip() for col in df.columns]
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    sections: list[dict] = []

    overview = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Duplicate rows removed": dup_count,
        "Total missing cells": int(df.isna().sum().sum()),
    }
    sections.append(
        {
            "title": "Dataset overview",
            "html": pd.DataFrame(list(overview.items()), columns=["Metric", "Value"]).to_html(
                classes="table table-bordered text-center align-middle", index=False
            ),
        }
    )

    if df.shape[1] > 0:
        dtypes = (
            pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": [str(t) for t in df.dtypes],
                    "Non-null": [int(df[c].notna().sum()) for c in df.columns],
                    "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
                }
            )
            .sort_values(["Type", "Column"])
            .reset_index(drop=True)
        )
        sections.append(
            {
                "title": "Columns & types",
                "html": dtypes.to_html(
                    classes="table table-striped table-bordered text-center align-middle", index=False
                ),
            }
        )

        missing_by_col = (
            df.isna()
            .sum()
            .rename("Missing")
            .to_frame()
            .assign(**{"Missing %": lambda x: (x["Missing"] / max(len(df), 1) * 100).round(2)})
            .reset_index(names="Column")
            .sort_values(["Missing", "Column"], ascending=[False, True])
        )
        if int(missing_by_col["Missing"].sum()) > 0:
            sections.append(
                {
                    "title": "Missing values by column",
                    "html": missing_by_col.to_html(
                        classes="table table-striped table-bordered text-center align-middle", index=False
                    ),
                }
            )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        num_desc = df[numeric_cols].describe().T.reset_index(names="Column")
        sections.append(
            {
                "title": "Numeric summary",
                "html": num_desc.to_html(
                    classes="table table-striped table-bordered text-center align-middle", index=False
                ),
            }
        )

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            # Build pairs of correlations excluding the diagonal.
            mask = np.eye(corr.shape[0], dtype=bool)
            pairs = corr.where(~mask).stack().reset_index()
            pairs = pairs.rename(columns={"level_0": "A", "level_1": "B", 0: "corr"})
            pairs["abs_corr"] = pairs["corr"].abs()
            top_corr = pairs.sort_values("abs_corr", ascending=False).head(10)[["A", "B", "corr"]]
            if not top_corr.empty:
                sections.append(
                    {
                        "title": "Top numeric correlations (absolute)",
                        "html": top_corr.to_html(
                            classes="table table-striped table-bordered text-center align-middle", index=False
                        ),
                    }
                )

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols[:10]:
        nunique = int(df[col].nunique(dropna=True))
        if nunique == 0:
            continue
        if nunique <= 50:
            top = (
                df[col]
                .astype("string")
                .fillna("<missing>")
                .value_counts()
                .head(10)
                .rename("Count")
                .to_frame()
                .reset_index(names=str(col))
            )
            sections.append(
                {
                    "title": f"Top values: {col}",
                    "html": top.to_html(
                        classes="table table-striped table-bordered text-center align-middle", index=False
                    ),
                }
            )

    # Candidate columns for chart picker UI.
    low_card_cat_cols = [
        c
        for c in cat_cols
        if int(df[c].nunique(dropna=True)) > 0 and int(df[c].nunique(dropna=True)) <= 50
    ][:20]
    numeric_candidate_cols = numeric_cols[:20]

    # If the CSV looks like a transaction dataset, add the richer financial analysis.
    transaction_cols = {"TransactionDate", "Quantity", "Price", "ProductID", "CustomerID", "ProductCategory"}
    if transaction_cols.issubset(set(df.columns)):
        tx = df.copy()
        tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
        tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce")
        tx["Price"] = pd.to_numeric(tx["Price"], errors="coerce")
        tx["TotalAmount"] = tx["Quantity"] * tx["Price"]

        summary_stats = {
            "Total Transactions": int(len(tx)),
            "Total Revenue": float(tx["TotalAmount"].sum(skipna=True)),
            "Average Transaction Value": round(float(tx["TotalAmount"].mean(skipna=True)), 2)
            if len(tx) > 0
            else 0.0,
            "Unique Customers": int(tx["CustomerID"].nunique()),
            "Unique Products": int(tx["ProductID"].nunique()),
        }
        sections.append(
            {
                "title": "Financial summary (detected transaction columns)",
                "html": pd.DataFrame(list(summary_stats.items()), columns=["Metric", "Value"]).to_html(
                    classes="table table-bordered text-center align-middle", index=False
                ),
            }
        )

        top_products = (
            tx.groupby("ProductID")["Quantity"].sum().sort_values(ascending=False).head(5).reset_index()
        )
        top_customers = (
            tx.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False).head(5).reset_index()
        )
        category_revenue = (
            tx.groupby("ProductCategory")["TotalAmount"].sum().sort_values(ascending=False).reset_index()
        )

        sections.append(
            {
                "title": "Top products",
                "html": top_products.to_html(
                    classes="table table-striped table-bordered text-center align-middle", index=False
                ),
            }
        )
        sections.append(
            {
                "title": "Top customers",
                "html": top_customers.to_html(
                    classes="table table-striped table-bordered text-center align-middle", index=False
                ),
            }
        )
        sections.append(
            {
                "title": "Revenue by category",
                "html": category_revenue.to_html(
                    classes="table table-striped table-bordered text-center align-middle", index=False
                ),
            }
        )

    return {
        "sections": sections,
        "chart_candidates": {
            "categorical_columns": low_card_cat_cols,
            "numeric_columns": numeric_candidate_cols,
        },
    }


def _pick_default_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@app.route("/chart/<filename>", methods=["GET"])
def chart(filename: str):
    # Render a chart for the uploaded file using Plotly.
    from plotly import express as px

    filepath = UPLOAD_FOLDER / filename
    df = load_data(filepath)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.drop_duplicates()

    chart_type = (request.args.get("chart_type") or "bar").strip().lower()
    x_col = (request.args.get("x_col") or "").strip()
    y_col = (request.args.get("y_col") or "").strip()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_card_cat_cols = [
        c for c in cat_cols if int(df[c].nunique(dropna=True)) > 0 and int(df[c].nunique(dropna=True)) <= 50
    ]

    # Sensible defaults if user didn't pick anything.
    if not x_col:
        x_col = _pick_default_column(df, low_card_cat_cols) or _pick_default_column(df, numeric_cols) or (
            df.columns[0] if len(df.columns) else ""
        )
    if not y_col:
        # Only set y_col when it makes sense (scatter/bar with numeric target, histogram uses x_col).
        if numeric_cols:
            y_col = numeric_cols[0]

    # ---- Chart generation ----
    plot_html = ""
    try:
        if chart_type == "pie":
            if x_col not in df.columns:
                raise ValueError("Select a valid x column for pie chart.")
            vc = (
                df[x_col].astype("string").fillna("<missing>").value_counts(dropna=False).head(10)
            )
            other = int(df.shape[0]) - int(vc.sum())
            if other > 0:
                vc.loc["Other"] = other
            plot_html = px.pie(vc.reset_index(name="Count"), names=x_col, values="Count").to_html(
                full_html=False, include_plotlyjs="cdn"
            )

        elif chart_type == "bar":
            if x_col not in df.columns:
                raise ValueError("Select a valid x column for bar chart.")
            if y_col and y_col in df.columns and y_col in numeric_cols:
                tmp = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10).reset_index()
                plot_html = px.bar(tmp, x=x_col, y=y_col).to_html(
                    full_html=False, include_plotlyjs="cdn"
                )
            else:
                vc = df[x_col].astype("string").fillna("<missing>").value_counts(dropna=False).head(10)
                tmp = vc.reset_index()
                tmp.columns = [x_col, "Count"]
                plot_html = px.bar(tmp, x=x_col, y="Count").to_html(
                    full_html=False, include_plotlyjs="cdn"
                )

        elif chart_type == "histogram":
            if x_col not in df.columns:
                raise ValueError("Select a valid numeric x column for histogram.")
            if x_col not in numeric_cols:
                raise ValueError("Histogram x column must be numeric.")
            plot_html = px.histogram(df, x=x_col).to_html(full_html=False, include_plotlyjs="cdn")

        elif chart_type == "scatter":
            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError("Select valid x and y numeric columns for scatter.")
            if x_col not in numeric_cols or y_col not in numeric_cols:
                raise ValueError("Scatter columns must be numeric.")
            tmp = df[[x_col, y_col]].dropna()
            plot_html = px.scatter(tmp, x=x_col, y=y_col, trendline="ols").to_html(
                full_html=False, include_plotlyjs="cdn"
            )

        elif chart_type == "line":
            # If a transaction date exists, prefer it; otherwise use dataframe index.
            date_col = None
            for candidate in ["TransactionDate", "Date", "date", "timestamp", "Timestamp"]:
                if candidate in df.columns:
                    date_col = candidate
                    break

            if y_col not in df.columns or y_col not in numeric_cols:
                raise ValueError("Line chart needs a numeric y column.")

            if date_col:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                tmp = pd.DataFrame({"date": dates, "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
                if tmp.empty:
                    raise ValueError("No valid dates/numeric values for line chart.")
                tmp = tmp.groupby("date")["y"].mean().reset_index().sort_values("date")
                plot_html = px.line(tmp, x="date", y="y").to_html(
                    full_html=False, include_plotlyjs="cdn"
                )
            else:
                y = pd.to_numeric(df[y_col], errors="coerce")
                tmp = pd.DataFrame({"x": range(len(y)), "y": y}).dropna()
                plot_html = px.line(tmp, x="x", y="y").to_html(
                    full_html=False, include_plotlyjs="cdn"
                )

        else:
            raise ValueError("Unsupported chart type.")
    except Exception as e:
        plot_html = f"<div class='alert alert-danger'>Chart error: {e}</div>"

    return render_template(
        "chart.html",
        filename=filename,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        plot_html=plot_html,
    )

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("No file selected.")
        elif not file.filename.lower().endswith((".csv", ".xls", ".xlsx", ".json", ".xml")):
            flash("Unsupported file format.")
        else:
            filepath = UPLOAD_FOLDER / file.filename
            file.save(filepath)
            return redirect(url_for("result", filename=file.filename))
    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    filepath = UPLOAD_FOLDER / filename
    try:
        data = analyze_data(filepath)
        return render_template("result.html", data=data, filename=filename)
    except Exception as e:
        return f"<h2>Error:</h2><pre>{e}</pre>"

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)