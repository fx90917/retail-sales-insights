from __future__ import annotations

import argparse
from pathlib import Path

from retail_sales_insights.analyzer import run_analysis
from retail_sales_insights.sample_data import generate_sample_orders


# 定义命令行接口参数及两个子命令。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail sales analytics and forecasting toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 生成模拟订单数据的子命令generate-sample。
    generate = subparsers.add_parser("generate-sample", help="Generate a synthetic order dataset.")
    generate.add_argument("--out", type=Path, default=Path("data/sample_orders.csv"))
    generate.add_argument("--rows", type=int, default=2400)
    generate.add_argument("--seed", type=int, default=42)

    # 执行分析的子命令analyze。
    analyze = subparsers.add_parser("analyze", help="Clean, segment, forecast, and chart retail sales data.")
    analyze.add_argument("--input", type=Path, default=Path("data/sample_orders.csv"))
    analyze.add_argument("--out", type=Path, default=Path("reports"))
    analyze.add_argument("--forecast-periods", type=int, default=6)
    return parser


# CLI主入口：读取参数并调用相应功能。
def main() -> None:
    parser = build_parser()
    # 把用户的命令行输入解析为参数对象。
    args = parser.parse_args()

    # 如果指定生成数据，则调用样例生成函数。
    if args.command == "generate-sample":
        path = generate_sample_orders(args.out, rows=args.rows, seed=args.seed)
        print(f"Generated sample data: {path}")
        return

    # 如果指定分析，则运行完整的数据处理管线。
    if args.command == "analyze":
        # 运行分析并取得输出文件路径。
        artifacts = run_analysis(args.input, args.out, forecast_periods=args.forecast_periods)
        print("Analysis complete:")
        # 把所有导出的报告路径显示在终端。
        for name, path in artifacts.__dict__.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

