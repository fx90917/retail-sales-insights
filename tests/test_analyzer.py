from __future__ import annotations

import unittest

import pandas as pd

from retail_sales_insights.analyzer import clean_orders, forecast_revenue, monthly_revenue, segment_customers


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "order_date": ["2025-01-02", "2025-01-03", "2025-02-04", "2025-03-05", "2025-04-06", "2025-05-07"],
            "customer_id": ["C1", "C1", "C2", "C3", "C4", "C5"],
            "region": ["East", "East", "West", "North", "South", "South"],
            "channel": ["Online", "Online", "Store", "Online", "Store", "Marketplace"],
            "category": ["Books", "Books", "Home", "Sports", "Beauty", "Books"],
            "product": ["Notebook", "Planner", "Lamp", "Mat", "Cream", "Book"],
            "quantity": [2, 1, 3, 2, 1, 4],
            "unit_price": [10, 20, 15, 25, 30, 12],
            "discount": [0.1, 0, 0.2, 0, 0.1, 0],
            "acquisition_cost": [5, 4, 8, 6, 7, 5],
        }
    )


class AnalyzerTests(unittest.TestCase):
    def test_clean_orders_adds_business_metrics(self) -> None:
        cleaned = clean_orders(sample_frame())

        self.assertTrue({"gross_sales", "net_sales", "profit", "order_month"}.issubset(cleaned.columns))
        self.assertEqual(cleaned.loc[0, "net_sales"], 18)
        self.assertTrue(cleaned["order_date"].is_monotonic_increasing)

    def test_segment_customers_returns_segments(self) -> None:
        cleaned = clean_orders(sample_frame())
        segments = segment_customers(cleaned)

        self.assertEqual(len(segments), cleaned["customer_id"].nunique())
        self.assertIn("segment", segments.columns)
        self.assertTrue(segments["rfm_score"].between(3, 12).all())

    def test_forecast_revenue_returns_requested_periods(self) -> None:
        cleaned = clean_orders(sample_frame())
        monthly = monthly_revenue(cleaned)
        forecast, mae = forecast_revenue(monthly, periods=3)

        self.assertEqual(len(forecast), 3)
        self.assertIn("forecast_revenue", forecast.columns)
        self.assertTrue(mae is None or mae >= 0)


if __name__ == "__main__":
    unittest.main()
