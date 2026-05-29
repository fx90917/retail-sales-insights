# Retail Sales Insights

一个中等难度、适合放到 GitHub 的 Python 数据分析项目。它使用可复现的零售订单数据完成数据清洗、关键指标统计、客户 RFM 分群、月度销售预测和图表导出。

## 项目亮点

- 可生成模拟零售订单数据，不依赖外部 API
- 自动清洗重复值、日期、数值字段和异常折扣
- 输出收入、利润、客单价、最佳品类、最佳区域等 KPI
- 使用 RFM 模型给客户分群
- 使用 `scikit-learn` 做月度销售预测
- 导出 CSV、JSON 和 PNG 图表，便于写报告或展示作品集
- 附带单元测试，适合作为 GitHub 项目继续扩展

## 目录结构

```text
retail_sales_insights/
├── data/
├── reports/
├── src/retail_sales_insights/
│   ├── analyzer.py
│   ├── cli.py
│   └── sample_data.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

生成示例数据：

```bash
retail-insights generate-sample --out data/sample_orders.csv --rows 2400
```

运行分析：

```bash
retail-insights analyze --input data/sample_orders.csv --out reports
```

运行测试：

```bash
python -m unittest discover -s tests
```

如果只想安装运行依赖，也可以使用：

```bash
pip install -r requirements.txt
```

## 输出文件

运行分析后，`reports/` 目录会生成：

- `summary.json`：核心 KPI 和客户分群数量
- `cleaned_orders.csv`：清洗后的订单明细
- `rfm_segments.csv`：客户 RFM 分群结果
- `monthly_forecast.csv`：未来月份销售预测
- `monthly_sales.png`：月度销售趋势图
- `category_revenue.png`：品类收入对比图

## 数据字段

输入 CSV 需要包含以下字段：

```text
order_id, order_date, customer_id, region, channel, category, product,
quantity, unit_price, discount, acquisition_cost
```

## 可以继续扩展的方向

- 增加 Streamlit 或 Flask 仪表盘
- 接入真实电商订单数据
- 增加 cohort 留存分析
- 比较更多预测模型，例如 Random Forest、XGBoost 或 Prophet
- 增加 GitHub Actions 自动运行测试
