from __future__ import annotations

import argparse
from pathlib import Path

from retail_sales_insights.analyzer import run_analysis
from retail_sales_insights.sample_data import generate_sample_orders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail sales analytics and forecasting toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-sample", help="Generate a synthetic order dataset.")
    generate.add_argument("--out", type=Path, default=Path("data/sample_orders.csv"))
    generate.add_argument("--rows", type=int, default=2400)
    generate.add_argument("--seed", type=int, default=42)

    analyze = subparsers.add_parser("analyze", help="Clean, segment, forecast, and chart retail sales data.")
    analyze.add_argument("--input", type=Path, default=Path("data/sample_orders.csv"))
    analyze.add_argument("--out", type=Path, default=Path("reports"))
    analyze.add_argument("--forecast-periods", type=int, default=6)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-sample":
        path = generate_sample_orders(args.out, rows=args.rows, seed=args.seed)
        print(f"Generated sample data: {path}")
        return

    if args.command == "analyze":
        artifacts = run_analysis(args.input, args.out, forecast_periods=args.forecast_periods)
        print("Analysis complete:")
        for name, path in artifacts.__dict__.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

