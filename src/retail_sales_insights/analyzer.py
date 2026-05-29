from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


REQUIRED_COLUMNS = {
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "channel",
    "category",
    "product",
    "quantity",
    "unit_price",
    "discount",
    "acquisition_cost",
}


@dataclass(frozen=True)
class AnalysisArtifacts:
    summary_json: Path
    cleaned_orders: Path
    rfm_segments: Path
    monthly_forecast: Path
    monthly_sales_chart: Path
    category_chart: Path


def load_orders(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return frame


def clean_orders(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.drop_duplicates(subset=["order_id"])
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["order_date", "customer_id", "category"])

    numeric_columns = ["quantity", "unit_price", "discount", "acquisition_cost"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=numeric_columns)
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] >= 0)]
    cleaned["discount"] = cleaned["discount"].clip(lower=0, upper=0.9)

    cleaned["gross_sales"] = cleaned["quantity"] * cleaned["unit_price"]
    cleaned["net_sales"] = cleaned["gross_sales"] * (1 - cleaned["discount"])
    cleaned["profit"] = cleaned["net_sales"] - cleaned["acquisition_cost"]
    cleaned["order_month"] = cleaned["order_date"].dt.to_period("M").dt.to_timestamp()
    return cleaned.sort_values("order_date").reset_index(drop=True)


def build_kpis(cleaned: pd.DataFrame) -> dict[str, float | int | str]:
    total_revenue = cleaned["net_sales"].sum()
    total_orders = cleaned["order_id"].nunique()
    return {
        "date_range": f"{cleaned['order_date'].min().date()} to {cleaned['order_date'].max().date()}",
        "orders": int(total_orders),
        "customers": int(cleaned["customer_id"].nunique()),
        "revenue": round(float(total_revenue), 2),
        "profit": round(float(cleaned["profit"].sum()), 2),
        "average_order_value": round(float(total_revenue / total_orders), 2),
        "best_category": str(cleaned.groupby("category")["net_sales"].sum().idxmax()),
        "best_region": str(cleaned.groupby("region")["net_sales"].sum().idxmax()),
    }


def segment_customers(cleaned: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = cleaned["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        cleaned.groupby("customer_id")
        .agg(
            recency=("order_date", lambda dates: (snapshot_date - dates.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("net_sales", "sum"),
        )
        .reset_index()
    )

    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["segment"] = np.select(
        [
            rfm["rfm_score"] >= 10,
            (rfm["r_score"] >= 3) & (rfm["f_score"] >= 3),
            rfm["r_score"] <= 2,
            rfm["m_score"] >= 4,
        ],
        ["Champions", "Loyal Customers", "At Risk", "Big Spenders"],
        default="Regular",
    )
    return rfm.sort_values(["rfm_score", "monetary"], ascending=False).reset_index(drop=True)


def monthly_revenue(cleaned: pd.DataFrame) -> pd.DataFrame:
    return (
        cleaned.groupby("order_month", as_index=False)
        .agg(revenue=("net_sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
        .sort_values("order_month")
    )


def forecast_revenue(monthly: pd.DataFrame, periods: int = 6) -> tuple[pd.DataFrame, float | None]:
    series = monthly.copy().reset_index(drop=True)
    series["month_index"] = np.arange(len(series))
    series["lag_1"] = series["revenue"].shift(1)
    series["lag_2"] = series["revenue"].shift(2)
    series["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(series["order_month"]).dt.month / 12)
    series["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(series["order_month"]).dt.month / 12)
    train = series.dropna().copy()

    if len(train) < 6:
        baseline = float(series["revenue"].tail(3).mean())
        future_months = pd.date_range(series["order_month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        return pd.DataFrame({"order_month": future_months, "forecast_revenue": baseline}), None

    features = ["month_index", "lag_1", "lag_2", "month_sin", "month_cos"]
    split = max(len(train) - 3, 1)
    model = LinearRegression()
    model.fit(train.loc[: split - 1, features], train.loc[: split - 1, "revenue"])
    validation_pred = model.predict(train.loc[split:, features])
    mae = float(mean_absolute_error(train.loc[split:, "revenue"], validation_pred)) if len(train.loc[split:]) else None

    model.fit(train[features], train["revenue"])
    history = series[["order_month", "revenue"]].copy()
    rows = []
    for step in range(1, periods + 1):
        next_month = history["order_month"].max() + pd.offsets.MonthBegin(1)
        revenue_values = history["revenue"].tail(2).to_list()
        feature_row = pd.DataFrame(
            {
                "month_index": [len(history)],
                "lag_1": [revenue_values[-1]],
                "lag_2": [revenue_values[-2]],
                "month_sin": [np.sin(2 * np.pi * next_month.month / 12)],
                "month_cos": [np.cos(2 * np.pi * next_month.month / 12)],
            }
        )
        prediction = max(0, float(model.predict(feature_row)[0]))
        rows.append({"order_month": next_month, "forecast_revenue": round(prediction, 2)})
        history = pd.concat([history, pd.DataFrame({"order_month": [next_month], "revenue": [prediction]})])

    return pd.DataFrame(rows), mae


def create_charts(cleaned: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_chart = output_dir / "monthly_sales.png"
    category_chart = output_dir / "category_revenue.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly["order_month"], monthly["revenue"], marker="o", linewidth=2.2, color="#2563eb")
    ax.set_title("Monthly Net Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(monthly_chart, dpi=160)
    plt.close(fig)

    category = cleaned.groupby("category")["net_sales"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    category.plot(kind="barh", ax=ax, color="#16a34a")
    ax.set_title("Revenue by Category")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(category_chart, dpi=160)
    plt.close(fig)

    return monthly_chart, category_chart


def run_analysis(input_path: Path, output_dir: Path, forecast_periods: int = 6) -> AnalysisArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = clean_orders(load_orders(input_path))
    rfm = segment_customers(cleaned)
    monthly = monthly_revenue(cleaned)
    forecast, mae = forecast_revenue(monthly, periods=forecast_periods)
    monthly_chart, category_chart = create_charts(cleaned, monthly, output_dir)

    cleaned_path = output_dir / "cleaned_orders.csv"
    rfm_path = output_dir / "rfm_segments.csv"
    forecast_path = output_dir / "monthly_forecast.csv"
    summary_path = output_dir / "summary.json"

    cleaned.to_csv(cleaned_path, index=False)
    rfm.to_csv(rfm_path, index=False)
    forecast.to_csv(forecast_path, index=False)

    summary = build_kpis(cleaned)
    summary["forecast_mae"] = round(mae, 2) if mae is not None else None
    summary["top_customer_segments"] = rfm["segment"].value_counts().to_dict()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return AnalysisArtifacts(
        summary_json=summary_path,
        cleaned_orders=cleaned_path,
        rfm_segments=rfm_path,
        monthly_forecast=forecast_path,
        monthly_sales_chart=monthly_chart,
        category_chart=category_chart,
    )

