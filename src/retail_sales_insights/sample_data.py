from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# 定义商品类别与每类商品名称。
PRODUCTS = {
    "Electronics": ["Wireless Mouse", "USB-C Hub", "Bluetooth Speaker", "Webcam"],
    "Home": ["Desk Lamp", "Air Purifier", "Storage Box", "Coffee Grinder"],
    "Sports": ["Yoga Mat", "Running Bottle", "Resistance Band", "Bike Light"],
    "Beauty": ["Face Serum", "Shampoo Set", "Hand Cream", "Travel Mirror"],
    "Books": ["Notebook", "Data Science Book", "Planner", "Sketch Pad"],
}

# 定义模拟销售地区。
REGIONS = ["East", "South", "West", "North"]
# 定义模拟销售渠道。
CHANNELS = ["Online", "Retail Store", "Marketplace"]


# 生成模拟零售订单，方便在没有真实业务数据时练习。
def generate_sample_orders(output_path: Path, rows: int = 2400, seed: int = 42) -> Path:
    """Create a realistic synthetic retail order dataset."""
    # 设置随机种子，保证相同参数时数据可重现。
    rng = np.random.default_rng(seed)
    # 生成2024至2025年期间的日期序列。
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    category_names = np.array(list(PRODUCTS.keys()))

    # 随机抽取每张订单的日期。
    sampled_dates = rng.choice(dates, size=rows)
    # 按照各商品类别的概率抽样。
    categories = rng.choice(category_names, p=[0.28, 0.22, 0.17, 0.18, 0.15], size=rows)
    # 在所属类别中随机选择商品。
    products = [rng.choice(PRODUCTS[category]) for category in categories]

    # 定义不同商品类别的参考单价。
    base_prices = {
        "Electronics": 65,
        "Home": 42,
        "Sports": 28,
        "Beauty": 24,
        "Books": 18,
    }
    # 使用正态分布生成有波动的商品价格。
    unit_prices = np.array(
        [max(5, rng.normal(base_prices[category], base_prices[category] * 0.25)) for category in categories]
    ).round(2)

    # 模拟每笔订单购买的件数。
    quantities = rng.poisson(lam=2.1, size=rows) + 1
    # 按概率模拟不同折扣比例。
    discounts = rng.choice([0, 0.05, 0.1, 0.15, 0.2], p=[0.48, 0.22, 0.16, 0.1, 0.04], size=rows)
    # 生成整单成本示例值，后续从折后收入直接扣除。
    acquisition_cost = rng.uniform(2.5, 16.0, size=rows).round(2)

    # 模拟客户ID：同一客户可能出现多次购买。
    customer_ids = [f"C{num:05d}" for num in rng.integers(1, 520, size=rows)]
    # 使用Pandas DataFrame组装订单数据。
    frame = pd.DataFrame(
        {
            "order_id": [f"O{num:06d}" for num in range(1, rows + 1)],
            "order_date": pd.to_datetime(sampled_dates).strftime("%Y-%m-%d"),
            "customer_id": customer_ids,
            "region": rng.choice(REGIONS, p=[0.29, 0.27, 0.24, 0.20], size=rows),
            "channel": rng.choice(CHANNELS, p=[0.50, 0.30, 0.20], size=rows),
            "category": categories,
            "product": products,
            "quantity": quantities,
            "unit_price": unit_prices,
            "discount": discounts,
            "acquisition_cost": acquisition_cost,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 按订单日期排序并输出CSV文件。
    frame.sort_values("order_date").to_csv(output_path, index=False)
    return output_path

