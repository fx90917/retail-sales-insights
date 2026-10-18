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


# 读取CSV并检查输入数据是否具备所有必要字段。
def load_orders(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return frame


# 数据清洗：去重、处理缺失值、校正类型、计算销售额。
def clean_orders(frame: pd.DataFrame) -> pd.DataFrame:
    # 复制DataFrame，避免直接修改原始输入。
    cleaned = frame.copy()
    # 以订单编号去重，避免重复计入销售额。
    cleaned = cleaned.drop_duplicates(subset=["order_id"])
    # 将文本日期转换为datetime；无效日期将变为NaT。
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    # 删除关键字段缺失的记录。
    cleaned = cleaned.dropna(subset=["order_date", "customer_id", "category"])

    numeric_columns = ["quantity", "unit_price", "discount", "acquisition_cost"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # 删除关键字段缺失的记录。
    cleaned = cleaned.dropna(subset=numeric_columns)
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] >= 0)]
    # 将异常折扣约束在0到90%之间。
    cleaned["discount"] = cleaned["discount"].clip(lower=0, upper=0.9)

    # 原价销售额=购买数量×单价。
    cleaned["gross_sales"] = cleaned["quantity"] * cleaned["unit_price"]
    # 折后收入=原价销售额×(1-折扣率)。
    cleaned["net_sales"] = cleaned["gross_sales"] * (1 - cleaned["discount"])
    # 订单利润=折后收入-订单成本；注意成本字段应为整笔订单成本。
    cleaned["profit"] = cleaned["net_sales"] - cleaned["acquisition_cost"]
    # 提取月份，统一使用该月第一天作为月份标识。
    cleaned["order_month"] = cleaned["order_date"].dt.to_period("M").dt.to_timestamp()
    return cleaned.sort_values("order_date").reset_index(drop=True)


# 计算整体经营指标（订单数、客户数、收入、利润、客单价等）。
def build_kpis(cleaned: pd.DataFrame) -> dict[str, float | int | str]:
    total_revenue = cleaned["net_sales"].sum()
    # nunique用于统计去重后的订单数量。
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


# 使用RFM模型对客户价值分群：最近购买、购买频率、消费金额。
def segment_customers(cleaned: pd.DataFrame) -> pd.DataFrame:
    # 用最新订单日期后一天作为RFM观察基准日。
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

    # 按最近购买时间分位数评分，越近期分数越高。
    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    # 按订单频率分位数评分，越常购买分数越高。
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    # 按消费额分位数评分，金额越高分数越高。
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    # 把三个1-4分相加，得到总分3-12分。
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    # 根据RFM分数判断冠军客户、忠诚客户和流失风险客户等。
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


# 将订单按月份汇总，得到每月收入、利润和订单数。
def monthly_revenue(cleaned: pd.DataFrame) -> pd.DataFrame:
    return (
        cleaned.groupby("order_month", as_index=False)
        .agg(revenue=("net_sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
        .sort_values("order_month")
    )


# 预测未来收入：数据充足用线性回归，数据少用最近三个月平均值。
def forecast_revenue(monthly: pd.DataFrame, periods: int = 6) -> tuple[pd.DataFrame, float | None]:
    series = monthly.copy().reset_index(drop=True)
    # 生成月份编号作为长期趋势特征。
    series["month_index"] = np.arange(len(series))
    # shift(1)提取上个月收入，构建时序滞后特征。
    series["lag_1"] = series["revenue"].shift(1)
    # shift(2)提取上上个月收入。
    series["lag_2"] = series["revenue"].shift(2)
    # 将月份编码为sin值，以表达季节周期。
    series["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(series["order_month"]).dt.month / 12)
    # 与sin特征配合表示一年循环。
    series["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(series["order_month"]).dt.month / 12)
    train = series.dropna().copy()

    # 数据不足时采用最近3个月平均值作为预测基线。
    if len(train) < 6:
        baseline = float(series["revenue"].tail(3).mean())
        future_months = pd.date_range(series["order_month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        return pd.DataFrame({"order_month": future_months, "forecast_revenue": baseline}), None

    # 模型输入：月份趋势、前两月收入、两个季节性特征。
    features = ["month_index", "lag_1", "lag_2", "month_sin", "month_cos"]
    split = max(len(train) - 3, 1)
    # 创建scikit-learn线性回归模型。
    model = LinearRegression()
    model.fit(train.loc[: split - 1, features], train.loc[: split - 1, "revenue"])
    # 使用留出的最后三期数据评估预测。
    validation_pred = model.predict(train.loc[split:, features])
    # MAE为平均绝对误差，衡量预测与真实收入的平均偏差。
    mae = float(mean_absolute_error(train.loc[split:, "revenue"], validation_pred)) if len(train.loc[split:]) else None

    # 使用全部可用历史数据重新训练最终模型。
    model.fit(train[features], train["revenue"])
    history = series[["order_month", "revenue"]].copy()
    rows = []
    # 逐月递推预测；前一个月的预测值可成为下一月的输入。
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
        # 将模型输出的负收入限制为零。
        prediction = max(0, float(model.predict(feature_row)[0]))
        rows.append({"order_month": next_month, "forecast_revenue": round(prediction, 2)})
        history = pd.concat([history, pd.DataFrame({"order_month": [next_month], "revenue": [prediction]})])

    return pd.DataFrame(rows), mae


# 生成按月销售折线图和产品品类收入柱状图。
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

    # 按商品品类汇总净销售收入，供柱状图使用。
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


# 整条流程入口：读取→清洗→统计→分群→预测→画图→保存。
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

    # 导出清洗后的CSV；index=False不写入行索引。
    cleaned.to_csv(cleaned_path, index=False)
    # 导出每位客户的RFM评分和分群。
    rfm.to_csv(rfm_path, index=False)
    # 导出未来月份的预测结果。
    forecast.to_csv(forecast_path, index=False)

    summary = build_kpis(cleaned)
    summary["forecast_mae"] = round(mae, 2) if mae is not None else None
    summary["top_customer_segments"] = rfm["segment"].value_counts().to_dict()
    # 将统计指标导出为JSON，便于其他系统读取。
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return AnalysisArtifacts(
        summary_json=summary_path,
        cleaned_orders=cleaned_path,
        rfm_segments=rfm_path,
        monthly_forecast=forecast_path,
        monthly_sales_chart=monthly_chart,
        category_chart=category_chart,
    )

