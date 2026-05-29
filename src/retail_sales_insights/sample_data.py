from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRODUCTS = {
    "Electronics": ["Wireless Mouse", "USB-C Hub", "Bluetooth Speaker", "Webcam"],
    "Home": ["Desk Lamp", "Air Purifier", "Storage Box", "Coffee Grinder"],
    "Sports": ["Yoga Mat", "Running Bottle", "Resistance Band", "Bike Light"],
    "Beauty": ["Face Serum", "Shampoo Set", "Hand Cream", "Travel Mirror"],
    "Books": ["Notebook", "Data Science Book", "Planner", "Sketch Pad"],
}

REGIONS = ["East", "South", "West", "North"]
CHANNELS = ["Online", "Retail Store", "Marketplace"]


def generate_sample_orders(output_path: Path, rows: int = 2400, seed: int = 42) -> Path:
    """Create a realistic synthetic retail order dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    category_names = np.array(list(PRODUCTS.keys()))

    sampled_dates = rng.choice(dates, size=rows)
    categories = rng.choice(category_names, p=[0.28, 0.22, 0.17, 0.18, 0.15], size=rows)
    products = [rng.choice(PRODUCTS[category]) for category in categories]

    base_prices = {
        "Electronics": 65,
        "Home": 42,
        "Sports": 28,
        "Beauty": 24,
        "Books": 18,
    }
    unit_prices = np.array(
        [max(5, rng.normal(base_prices[category], base_prices[category] * 0.25)) for category in categories]
    ).round(2)

    quantities = rng.poisson(lam=2.1, size=rows) + 1
    discounts = rng.choice([0, 0.05, 0.1, 0.15, 0.2], p=[0.48, 0.22, 0.16, 0.1, 0.04], size=rows)
    acquisition_cost = rng.uniform(2.5, 16.0, size=rows).round(2)

    customer_ids = [f"C{num:05d}" for num in rng.integers(1, 520, size=rows)]
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
    frame.sort_values("order_date").to_csv(output_path, index=False)
    return output_path

