"""Build the canonical Data Analytics artifact for the detailed FRTBOT report.

Input datasets are produced by ``scripts/export_detailed_backtest.py``.  The
resulting ``artifact.json`` is consumed by the Data Analytics portable report
builder, which creates one self-contained local HTML file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "detailed_report"
DATA_DIR = REPORT_DIR / "data"
GENERATED_AT = "2026-08-04T16:00:00+07:00"

DISPLAY_NAMES = {
    "base": "FRTBOT Base",
    "zero": "FRTBOT Zero cost",
    "optimistic": "FRTBOT Optimistic",
    "severe": "FRTBOT Severe",
    "equal_weight": "Equal-weight 4 markets",
    "buy_hold_CN": "Buy & hold China",
    "buy_hold_EU": "Buy & hold Europe",
    "buy_hold_TH": "Buy & hold Thailand",
    "buy_hold_US": "Buy & hold United States",
    "cash": "Cash THB 1.25%",
    "tree": "Tree model",
    "mlp": "MLP neural network",
    "transparent_trend": "Transparent trend",
    "linear_ridge": "Ridge regression",
    "rule_momentum": "Momentum rule",
    "equal_weight_score": "Equal score",
    "ensemble": "Frozen ensemble",
}


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DATA_DIR / name, index=False)


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    tables_used: list[str],
    filters: list[str] | None = None,
    metric_definitions: list[str] | None = None,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "description": description,
            "executed_at": GENERATED_AT,
            "tables_used": tables_used,
            "filters": filters or [],
            "metric_definitions": metric_definitions or [],
        },
    }


def _chart(
    chart_id: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    source_id: str,
    x: dict,
    y: dict,
    *,
    color: dict | None = None,
    value_format: str | None = None,
    intent: str | None = None,
    tooltip: list[dict] | None = None,
) -> dict:
    encodings = {"x": x, "y": y}
    if color:
        encodings["color"] = color
    if tooltip:
        encodings["tooltip"] = tooltip
    result = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": encodings,
        "layout": "full",
    }
    if value_format:
        result["valueFormat"] = value_format
    if intent:
        result["intent"] = intent
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(DATA_DIR / "scenario_summary.csv")
    benchmarks = pd.read_csv(DATA_DIR / "benchmark_summary.csv")
    folds = pd.read_csv(DATA_DIR / "fold_summary.csv")
    model_folds = pd.read_csv(DATA_DIR / "model_metrics_by_fold.csv")
    daily = pd.read_csv(DATA_DIR / "daily_performance.csv", parse_dates=["date"])
    daily_weights = pd.read_csv(DATA_DIR / "daily_weights.csv", parse_dates=["date"])
    rebalance_weights = pd.read_csv(DATA_DIR / "rebalance_weights.csv")
    audit = pd.read_csv(DATA_DIR / "data_audit.csv")
    anomalies = pd.read_csv(DATA_DIR / "price_anomalies.csv")
    attribution = pd.read_csv(DATA_DIR / "country_attribution.csv")
    feature_catalog = pd.read_csv(DATA_DIR / "feature_catalog.csv")

    scenarios["display_name"] = scenarios["series"].map(DISPLAY_NAMES)
    benchmarks["display_name"] = benchmarks["series"].map(DISPLAY_NAMES)
    scenarios["max_drawdown_abs"] = scenarios["max_drawdown_net"].abs()
    benchmarks["max_drawdown_abs"] = benchmarks["max_drawdown_net"].abs()
    performance_summary = pd.concat(
        [scenarios.assign(group="FRTBOT cost scenario"), benchmarks.assign(group="Benchmark")],
        ignore_index=True,
        sort=False,
    )
    _write_csv(performance_summary, "performance_summary.csv")

    base = scenarios.loc[scenarios["series"] == "base"].iloc[0]
    equal_weight = benchmarks.loc[benchmarks["series"] == "equal_weight"].iloc[0]
    zero = scenarios.loc[scenarios["series"] == "zero"].iloc[0]
    severe = scenarios.loc[scenarios["series"] == "severe"].iloc[0]
    headline = pd.DataFrame(
        [
            {
                "base_cagr": base["cagr_net"],
                "base_cumulative": base["cumulative_return_net"],
                "base_sharpe": base["sharpe_net"],
                "base_max_drawdown": base["max_drawdown_net"],
                "equal_weight_cagr": equal_weight["cagr_net"],
                "cagr_gap_vs_equal_weight": base["cagr_net"] - equal_weight["cagr_net"],
                "base_cost_drag": base["cost_drag"],
                "severe_cagr": severe["cagr_net"],
            }
        ]
    )
    _write_csv(headline, "headline_metrics.csv")

    equity_columns = {
        "strategy_base_equity": "FRTBOT Base",
        "benchmark_equal_weight_equity": "Equal-weight 4 markets",
        "benchmark_cash_equity": "Cash THB 1.25%",
    }
    month_end = daily.set_index("date").resample("ME").last().reset_index()
    equity_rows = []
    for column, display_name in equity_columns.items():
        for row in month_end[["date", column]].itertuples(index=False):
            equity_rows.append(
                {
                    "date": row.date.date().isoformat(),
                    "series": display_name,
                    "growth_index": float(getattr(row, column)),
                }
            )
    equity_monthly = pd.DataFrame(equity_rows)
    _write_csv(equity_monthly, "equity_curve_monthly.csv")

    fold_rows = []
    for row in folds.itertuples(index=False):
        label = f"Fold {row.fold}: {row.test_start} to {row.test_end}"
        fold_rows.extend(
            [
                {
                    "fold": int(row.fold),
                    "fold_label": label,
                    "series": "FRTBOT Base",
                    "cumulative_return": float(row.cumulative_return_net),
                    "cagr": float(row.cagr_net),
                },
                {
                    "fold": int(row.fold),
                    "fold_label": label,
                    "series": "Equal-weight 4 markets",
                    "cumulative_return": float(row.equal_weight_cumulative_return_net),
                    "cagr": float(row.equal_weight_cagr_net),
                },
            ]
        )
    fold_comparison = pd.DataFrame(fold_rows)
    _write_csv(fold_comparison, "fold_comparison.csv")

    model_average = (
        model_folds.groupby("model", as_index=False)
        .agg(
            rank_ic=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            top_quintile_hit_rate=("top_quintile_hit_rate", "mean"),
            scored_rows=("n_scored_rows", "sum"),
            folds=("fold", "nunique"),
        )
        .sort_values("rank_ic", ascending=False)
    )
    model_average["display_name"] = model_average["model"].map(DISPLAY_NAMES)
    _write_csv(model_average, "model_metrics_average.csv")

    allocation_daily = rebalance_weights.rename(
        columns={"signal_date": "date", "target_weight": "weight"}
    )[["fold", "date", "asset", "weight"]].copy()
    allocation_summary_rows = []
    contribution_map = attribution.set_index("market")["gross_return_contribution"].to_dict()
    for asset in ["CN", "EU", "TH", "US", "CASH_THB"]:
        values = daily_weights[asset]
        allocation_summary_rows.append(
            {
                "asset": asset,
                "average_weight": float(values.mean()),
                "maximum_weight": float(values.max()),
                "days_nonzero_share": float((values > 0).mean()),
                "gross_return_contribution": contribution_map.get(asset),
            }
        )
    allocation_summary = pd.DataFrame(allocation_summary_rows)
    _write_csv(allocation_summary, "allocation_summary.csv")

    audit["coverage"] = audit["business_day_coverage_ratio"]
    audit["anomaly_count"] = audit["price_anomaly_dates"].fillna("[]").map(
        lambda value: 0 if value == "[]" else value.count("datetime.date")
    )
    _write_csv(audit, "data_quality_summary.csv")

    metric_definitions = pd.DataFrame(
        [
            {"metric": "CAGR", "finance_language": "Compound Annual Growth Rate", "plain_thai": "อัตราโตเฉลี่ยต่อปีแบบทบต้น; ไม่ใช่ค่าเฉลี่ยเลขคณิต"},
            {"metric": "Annualized volatility", "finance_language": "ส่วนเบี่ยงเบนมาตรฐานรายวัน × √252", "plain_thai": "ความแกว่งของผลตอบแทนต่อปี; สูงแปลว่าพอร์ตเหวี่ยงมาก"},
            {"metric": "Sharpe ratio", "finance_language": "ผลตอบแทนส่วนเกินเฉลี่ย ÷ volatility", "plain_thai": "ผลตอบแทนที่ได้ต่อความเสี่ยงรวม; ในรายงานใช้ cash 1.25% เป็น risk-free proxy"},
            {"metric": "Sortino ratio", "finance_language": "ผลตอบแทนส่วนเกิน ÷ downside deviation", "plain_thai": "คล้าย Sharpe แต่ลงโทษเฉพาะความผันผวนด้านลบ"},
            {"metric": "Maximum drawdown", "finance_language": "ขาดทุนสูงสุดจากจุดสูงสุดถึงจุดต่ำสุด", "plain_thai": "ช่วงที่พอร์ตย่อลึกที่สุดก่อนฟื้น"},
            {"metric": "Rank IC", "finance_language": "ค่าเฉลี่ย Spearman correlation ระหว่าง score กับผลตอบแทนล่วงหน้า", "plain_thai": "คะแนนโมเดลเรียงลำดับตลาดถูกทิศแค่ไหน; 0 คือไม่มีความสัมพันธ์เชิงอันดับ"},
            {"metric": "Top-quintile hit rate", "finance_language": "สัดส่วนวันที่ตัวเลือกอันดับสูงสุดตรงกับผู้ชนะจริงใน top 20%", "plain_thai": "โมเดลทายตลาดตัวบนสุดถูกบ่อยแค่ไหน"},
            {"metric": "Cost drag", "finance_language": "ผลตอบแทนสะสม gross − net", "plain_thai": "กำไรที่หายไปเพราะค่าคอม สเปรด และ slippage ตามสมมุติฐาน"},
        ]
    )
    _write_csv(metric_definitions, "metric_definitions.csv")

    engineering = pd.DataFrame(
        [
            {"layer": "Data", "module": "frtbot.data", "role": "Yahoo Finance cache, immutable raw partitions, normalized Parquet, FX alignment, quality gate"},
            {"layer": "Features", "module": "frtbot.features", "role": "Trend, momentum, volatility, drawdown, liquidity, THB return, cross-market context"},
            {"layer": "Labels", "module": "frtbot.labels", "role": "21-trading-day forward THB excess return and positive-return class"},
            {"layer": "Models", "module": "frtbot.models", "role": "Tree 40%, MLP 30%, transparent trend 30%, plus ridge/rule/equal-score baselines"},
            {"layer": "Portfolio", "module": "frtbot.portfolio", "role": "Positive-return gate, top 3 markets, inverse 63-day vol, 40% cap, cash residual"},
            {"layer": "Backtest", "module": "frtbot.backtest", "role": "Expanding walk-forward, 21-day embargo, next-bar execution, monthly rebalance, four cost regimes"},
            {"layer": "Reporting", "module": "frtbot.reporting", "role": "Portfolio metrics, ML metrics, data audit, attribution, exported evidence tables"},
            {"layer": "Verification", "module": "tests/", "role": "73 deterministic tests covering leakage, FX, costs, constraints, config, models and data quality"},
        ]
    )
    _write_csv(engineering, "engineering_components.csv")

    dataset_inventory = []
    descriptions = {
        "daily_performance.csv": "Daily OOS returns, equity curves and drawdowns for strategy and benchmarks",
        "daily_weights.csv": "Daily implemented weights for CN, EU, TH, US and THB cash",
        "rebalance_weights.csv": "Every target weight emitted at monthly signal dates",
        "scenario_summary.csv": "All portfolio metrics under four transaction-cost assumptions",
        "benchmark_summary.csv": "Equal-weight, cash and per-market buy-and-hold comparison",
        "fold_summary.csv": "Fold-level base strategy and equal-weight results",
        "model_metrics_by_fold.csv": "Rank IC and top-quintile hit rate by model and fold",
        "data_audit.csv": "Coverage, gaps, staleness and anomaly flags for markets and FX",
        "model_panel.parquet": "Full long-format feature and label panel (date × market)",
        "feature_catalog.csv": "Feature types, populated rows and null rates",
    }
    for name, description in descriptions.items():
        path = DATA_DIR / name
        if path.suffix == ".csv":
            row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
        else:
            row_count = len(pd.read_parquet(path))
        dataset_inventory.append(
            {
                "file": name,
                "rows": row_count,
                "size_bytes": path.stat().st_size,
                "description": description,
            }
        )
    dataset_inventory_frame = pd.DataFrame(dataset_inventory)
    _write_csv(dataset_inventory_frame, "dataset_inventory.csv")

    sources = [
        _source(
            "performance_export",
            "Walk-forward portfolio and benchmark export",
            "data/performance_summary.csv",
            "Re-runs all five chronological test folds and aggregates daily THB returns into portfolio metrics.",
            ["data/daily_performance.csv", "data/scenario_summary.csv", "data/benchmark_summary.csv"],
            ["OOS dates 2022-06-01 to 2026-07-30", "JP excluded by >40% daily-return anomaly gate", "THB base currency"],
            [
                "CAGR=(product(1+r))^(252/N)-1",
                "Sharpe=mean(daily return-cash/252)/stdev × sqrt(252)",
                "Cost drag=cumulative gross return-cumulative net return",
            ],
        ),
        _source(
            "fold_export",
            "Fold-by-fold walk-forward export",
            "data/fold_summary.csv",
            "Compares the base strategy with the equal-weight benchmark on identical fold test windows.",
            ["data/fold_summary.csv", "data/fold_comparison.csv"],
            ["8y train / 2y validation / 1y test / 1y step", "21-trading-day embargo"],
        ),
        _source(
            "model_export",
            "Model metrics by walk-forward fold",
            "data/model_metrics_by_fold.csv",
            "Computes Rank IC and top-quintile hit rate for every model on each test fold.",
            ["data/model_metrics_by_fold.csv", "data/model_metrics_average.csv"],
        ),
        _source(
            "allocation_export",
            "Portfolio weights and contribution export",
            "data/daily_weights.csv",
            "Preserves implemented daily weights, monthly targets and approximate country gross-return contribution.",
            ["data/daily_weights.csv", "data/rebalance_weights.csv", "data/country_attribution.csv"],
        ),
        _source(
            "data_audit_export",
            "Market and FX data audit",
            "data/data_quality_summary.csv",
            "Profiles locally cached real Yahoo Finance series for coverage, gaps, staleness and implausible returns.",
            ["data/data_audit.csv", "data/price_anomalies.csv"],
            ["Retrieval snapshot 2026-08-04", "Extreme-return threshold 40% in one session"],
        ),
        _source(
            "method_export",
            "Feature, metric and engineering definitions",
            "data/metric_definitions.csv",
            "Combines the implemented feature catalog, configured research protocol and repository component map.",
            ["data/feature_catalog.csv", "data/metric_definitions.csv", "data/engineering_components.csv", "data/run_manifest.csv"],
        ),
        _source(
            "inventory_export",
            "Detailed local dataset inventory",
            "data/dataset_inventory.csv",
            "Lists the detailed supporting datasets shipped beside the self-contained report.",
            ["data/dataset_inventory.csv"],
        ),
    ]

    summary_text = (
        "## Technical Summary\n\n"
        "- **ผลตอบแทนเป็นบวก แต่ยังไม่เกิด alpha เหนือกฎง่าย:** FRTBOT Base ทำ CAGR "
        f"**{base['cagr_net']:.2%}**, ผลตอบแทนสะสม **{base['cumulative_return_net']:.2%}**, "
        f"Sharpe **{base['sharpe_net']:.2f}** และ Max Drawdown **{base['max_drawdown_net']:.2%}**; "
        f"equal-weight ทำ CAGR **{equal_weight['cagr_net']:.2%}**, Sharpe **{equal_weight['sharpe_net']:.2f}** "
        f"และ Max Drawdown **{equal_weight['max_drawdown_net']:.2%}** บนช่วง OOS เดียวกัน\n"
        f"- **ต้นทุนมีผลจริง:** Zero-cost สะสม **{zero['cumulative_return_net']:.2%}** เทียบกับ Base "
        f"**{base['cumulative_return_net']:.2%}** หรือ cost drag **{base['cost_drag']:.2%}** ของเงินตั้งต้น; "
        f"Severe-cost ยังมี CAGR **{severe['cagr_net']:.2%}**\n"
        "- **ผลไม่เสถียรทุก fold:** Fold แรกขาดทุน ขณะที่ Fold 3 เด่นมาก; final fold มีเพียง 44 วันซื้อขาย "
        "จึงห้ามอ่าน CAGR ของ fold นั้นเทียบกับปีเต็มแบบตรง ๆ\n"
        "- **สถานะคำตัดสิน:** เหมาะสำหรับ research discussion แบบ **Share with caveats** แต่ยังไม่พร้อมใช้ตัดสินใจลงทุนจริง "
        "เพราะแพ้ equal-weight, ใช้ country proxy เพียง 4 ตลาด และยังไม่มี stock-level 10–15 ตัว"
    )

    blocks = [
        {"id": "title", "type": "markdown", "body": "# FRTBOT: รายงาน Backtest เชิงการเงินและวิศวกรรม"},
        {"id": "technical_summary", "type": "markdown", "body": summary_text},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["base_cagr", "base_cumulative", "base_sharpe", "base_drawdown", "equal_weight_cagr"]},
        {"id": "growth_heading", "type": "markdown", "body": "## เงินเติบโต แต่เส้น equal-weight ยังนำ\n\nกราฟใช้ดัชนีเริ่มต้น 1.00 และเก็บค่าปลายเดือนเพื่อเห็นรูปทรงชัดเจน เส้น FRTBOT Base อยู่เหนือ cash แต่จบต่ำกว่า equal-weight; benchmark รายประเทศทั้งหมดแสดงในส่วนถัดไป นี่แปลว่าโมเดลสร้าง absolute return ได้ แต่ยังพิสูจน์ relative alpha ไม่สำเร็จ"},
        {"id": "equity_chart", "type": "chart", "chartId": "equity_curve"},
        {"id": "cost_heading", "type": "markdown", "body": "## ต้นทุนลดผลตอบแทนโดยแทบไม่ลดความผันผวน\n\nจาก Zero ไป Base ผลตอบแทนสะสมหาย 3.59 จุดเปอร์เซ็นต์ ขณะที่ annualized volatility เกือบคงเดิม เพราะค่าธรรมเนียมถูกหักตอน rebalance ไม่ได้เปลี่ยน exposure หลักของพอร์ต ดังนั้น turnover เป็นตัวแปรเศรษฐศาสตร์ที่ต้องควบคุม ไม่ใช่แค่รายละเอียด implementation"},
        {"id": "cost_chart", "type": "chart", "chartId": "cost_sensitivity"},
        {"id": "scenario_table", "type": "table", "tableId": "scenario_detail"},
        {"id": "benchmark_heading", "type": "markdown", "body": "## Benchmark แบบง่ายชนะในมิติ risk-adjusted\n\nEqual-weight ให้ CAGR สูงกว่า 0.92 จุดเปอร์เซ็นต์, Sharpe สูงกว่า 0.14 และ drawdown ตื้นกว่าประมาณ 2.07 จุดเปอร์เซ็นต์ ขณะที่ Europe buy-and-hold ให้ CAGR สูงสุดแต่รับ volatility ราว 20%; จึงต้องแยกคำว่า ‘ผลตอบแทนสูง’ ออกจาก ‘ผลตอบแทนต่อความเสี่ยงดี’"},
        {"id": "benchmark_chart", "type": "chart", "chartId": "benchmark_cagr"},
        {"id": "benchmark_table", "type": "table", "tableId": "benchmark_detail"},
        {"id": "fold_heading", "type": "markdown", "body": "## ความสม่ำเสมอระหว่างเวลาเป็นจุดอ่อนหลัก\n\nFRTBOT ชนะ equal-weight ใน Fold 1 และช่วงสั้น Fold 4 แต่แพ้ใน Fold 0, Fold 2 และ Fold 3; Fold 0 ขาดทุน 6.98% ขณะที่ equal-weight บวก 2.18% ผลรวมบวกของทั้งระบบจึงพึ่งพาช่วงแข็งแรงบางช่วงมากกว่าการชนะสม่ำเสมอ"},
        {"id": "fold_chart", "type": "chart", "chartId": "fold_comparison"},
        {"id": "fold_table", "type": "table", "tableId": "fold_detail"},
        {"id": "model_heading", "type": "markdown", "body": "## Signal ranking มีข้อมูล แต่ยังแปลงเป็น portfolio alpha ไม่พอ\n\nFrozen ensemble มี Rank IC เฉลี่ยสูงสุดราว 0.12 ซึ่งหมายถึงความสัมพันธ์เชิงอันดับเป็นบวกแต่ไม่แรง ส่วน MLP มี top-quintile hit rate สูงสุดราว 0.34 การที่ metric ระดับโมเดลดูบวกแต่พอร์ตยังแพ้ equal-weight ชี้ว่าปัญหาอาจอยู่ที่ eligibility gate, การแปลง score เป็นน้ำหนัก, turnover, regime dependence หรือ sample ขนาดเล็กเพียง 4 ตลาดต่อวัน"},
        {"id": "model_chart", "type": "chart", "chartId": "model_rank_ic"},
        {"id": "model_table", "type": "table", "tableId": "model_detail"},
        {"id": "allocation_heading", "type": "markdown", "body": "## พอร์ตกระจาย 4 ตลาดและถือเงินสดเฉลี่ยราว 11.9%\n\nกฎเลือกสูงสุด 3 ประเทศต่อเดือนและจำกัดประเทศละ 40% ทำให้น้ำหนักสลับระหว่าง CN, EU, TH และ US ส่วน CASH_THB รับส่วนที่เหลือ Contribution เป็นการประมาณแบบผลรวม weight × return จึงใช้ดูทิศทาง ไม่ใช่ Brinson attribution แบบเต็ม"},
        {"id": "allocation_chart", "type": "chart", "chartId": "allocation_mix"},
        {"id": "allocation_table", "type": "table", "tableId": "allocation_detail"},
        {"id": "data_heading", "type": "markdown", "body": "## ข้อมูลครบด้าน coverage แต่ Japan ไม่ผ่าน plausibility gate\n\nตลาดและ FX ทั้ง 9 series ดาวน์โหลดได้จริงและครอบคลุม 2012–2026 แต่ 1306.T มี daily move ผิดธรรมชาติ 3 วัน จึงถูกตัดออกจาก model/backtest แบบ fail-loudly ข้อมูล ‘โหลดได้’ จึงไม่เท่ากับ ‘ใช้ลงทุนได้’ โดยเฉพาะ corporate action จาก free feed"},
        {"id": "coverage_chart", "type": "chart", "chartId": "data_coverage"},
        {"id": "audit_table", "type": "table", "tableId": "data_audit"},
        {"id": "scope_heading", "type": "markdown", "body": "## Scope, หน่วยวัด และคำศัพท์\n\nประชากรที่วิเคราะห์คือ country ETF/index proxies ในฐานเงิน THB ช่วง OOS 1 มิ.ย. 2022–30 ก.ค. 2026 รวม 1,084 trading days ตัวเลขหลักเป็น net of costs ตาม scenario; cash 1.25% เป็นสมมุติฐานคงที่ ไม่ใช่ market series จริง และผลลัพธ์ทั้งหมดเป็น paper research ไม่ใช่ realized P&L"},
        {"id": "definitions_table", "type": "table", "tableId": "metric_definitions"},
        {"id": "method_heading", "type": "markdown", "body": "## Methodology และ data lineage\n\nPipeline คือ Parquet cache → THB conversion → 36 feature columns → 21-day forward excess-return label → expanding walk-forward → Tree/MLP/Trend ensemble → positive-return gate → inverse-volatility weights → next-bar backtest ค่า scaler/model fit จาก train เท่านั้น มี embargo 21 วัน และ deterministic seed 42 เพื่อป้องกัน look-ahead และลด nondeterminism"},
        {"id": "engineering_table", "type": "table", "tableId": "engineering_components"},
        {"id": "limitations", "type": "markdown", "body": "## ข้อจำกัด ความไม่แน่นอน และ robustness gaps\n\n- **ยังไม่ใช่พอร์ตหุ้น 10–15 ตัว:** ปัจจุบันเป็น country proxy และเลือกสูงสุด 3 ประเทศ; M3 stock adapters กับ M4 sequence model ยังไม่เริ่ม\n- **Benchmark และ fold stitching ยังควรยกระดับ:** รายงานนี้เพิ่ม benchmark ภายนอก README เดิมแล้ว แต่ backtest หลักรันแต่ละ fold แยกและเริ่มจาก cash ก่อนสัญญาณเดือนแรก จึงยังไม่ใช่ continuous holdings ledger ที่ carry position ข้าม fold\n- **Execution approximation:** สัญญาณจาก close วันที่ t มีผลต่อ close-to-close return ของ t+1 ไม่ใช่ราคาตลาดเปิดจริง\n- **Single free provider:** Yahoo Finance ไม่มี point-in-time membership, delisted-stock coverage หรือ corporate-action SLA ที่พอสำหรับ production\n- **Cross-exchange clock:** การเปรียบเทียบวันเดียวกันเป็น daily-date approximation ไม่ได้ synchronize ตามเวลาปิดจริงทุกตลาด\n- **Sample เล็ก:** เหลือ 4 ตลาดต่อวัน ทำให้ cross-sectional metric noisy; final fold มีเพียง 44 trading days\n- **ยังขาด uncertainty layer:** ไม่มี confidence interval, bootstrap, Deflated Sharpe หรือ Probability of Backtest Overfitting\n- **Metric ที่ defer:** Brier/calibration และ stability by regime รอ stock-level cross-section ที่ใหญ่ขึ้น"},
        {"id": "next_steps", "type": "markdown", "body": "## Recommended next steps\n\n1. ทำ benchmark ให้เป็น first-class ใน `run_full_backtest.py` และเพิ่ม continuous OOS holdings ledger ที่ carry น้ำหนัก/ต้นทุนข้าม fold\n2. เพิ่ม bootstrap confidence interval, fold/regime attribution และ sensitivity grid สำหรับ top-N, cap, horizon และ ensemble weights โดยห้ามเลือกค่าจาก final test\n3. เปลี่ยน Japan feed เป็น corporate-action-aware source หรือใช้ proxy สำรองที่ผ่านกติกา config+audit อย่างโปร่งใส\n4. เริ่ม M3 stock-level ทีละตลาดจากแหล่ง point-in-time ที่ดีที่สุด; เป้าหมายจำนวนหุ้นต้องเป็นผลจาก eligibility/constraints ไม่ใช่บังคับให้ครบ 10–15 ตัว\n5. อนุญาต TCN/1D-CNN เข้า shadow research หลัง tree/MLP pipeline มี benchmark และ robustness gate ครบเท่านั้น"},
        {"id": "further_questions", "type": "markdown", "body": "## Further questions\n\n- จะใช้ data provider ใดสำหรับ historical constituents, delisted securities และ corporate actions ของแต่ละประเทศ?\n- เป้าหมายหลักคือชนะ equal-weight ด้าน CAGR, Sharpe, drawdown หรือหลาย objective พร้อมกัน?\n- จะยอมรับ turnover/cost budget สูงสุดเท่าไร และต้องการถือเงินสดขั้นต่ำหรือไม่?\n- สำหรับพอร์ตหุ้น 10–15 ตัว จะเริ่มตลาดใดก่อนเพื่อให้ survivorship bias ต่ำสุด?"},
        {"id": "inventory_heading", "type": "markdown", "body": "## ชุดข้อมูลรายละเอียดที่ส่งมอบ\n\nตารางนี้ชี้ไปยังไฟล์ local ที่ใช้ตรวจซ้ำได้ ทั้ง daily return/equity, weights, fold/model metrics, audit และ full model panel รายงาน HTML เป็นไฟล์เดียวแบบ self-contained แต่ข้อมูลดิบเชิงวิเคราะห์ยังถูกแยกไว้เพื่อเปิดใน Jupyter, pandas หรือ Excel"},
        {"id": "inventory_table", "type": "table", "tableId": "dataset_inventory"},
    ]

    cards = [
        {"id": "base_cagr", "description": "ผลตอบแทนทบต้นต่อปีของ Base cost scenario", "dataset": "headline", "sourceId": "performance_export", "metrics": [{"label": "Base CAGR", "field": "base_cagr", "format": "percent"}, {"label": "vs equal-weight", "field": "cagr_gap_vs_equal_weight", "format": "percent", "signed": True}]},
        {"id": "base_cumulative", "description": "การเติบโตสุทธิของเงินตั้งต้นตลอด OOS", "dataset": "headline", "sourceId": "performance_export", "metrics": [{"label": "Cumulative return", "field": "base_cumulative", "format": "percent"}, {"label": "cost drag", "field": "base_cost_drag", "format": "percent", "signed": True}]},
        {"id": "base_sharpe", "description": "Excess return ต่อ volatility โดยใช้ cash 1.25%", "dataset": "headline", "sourceId": "performance_export", "metrics": [{"label": "Sharpe ratio", "field": "base_sharpe", "format": "number"}]},
        {"id": "base_drawdown", "description": "การย่อลึกสุดจาก peak ถึง trough", "dataset": "headline", "sourceId": "performance_export", "metrics": [{"label": "Max drawdown", "field": "base_max_drawdown", "format": "percent"}]},
        {"id": "equal_weight_cagr", "description": "CAGR ของ benchmark น้ำหนักเท่ากัน 4 ตลาด", "dataset": "headline", "sourceId": "performance_export", "metrics": [{"label": "Equal-weight CAGR", "field": "equal_weight_cagr", "format": "percent"}]},
    ]

    charts = [
        _chart("equity_curve", "ดัชนีการเติบโตของเงินลงทุน", "ค่าปลายเดือน; เริ่มต้นที่ 1.00, ฐานเงิน THB", "line", "equity_monthly", "performance_export", {"field": "date", "type": "temporal", "label": "Month"}, {"field": "growth_index", "type": "quantitative", "label": "Growth index"}, color={"field": "series", "type": "nominal", "label": "Series"}, value_format="number", intent="trend", tooltip=[{"field": "growth_index", "type": "quantitative", "label": "Growth index"}]),
        _chart("cost_sensitivity", "CAGR ภายใต้ต้นทุน 4 ระดับ", "Zero, optimistic, base และ severe; bar เริ่มจากศูนย์", "bar", "scenarios", "performance_export", {"field": "display_name", "type": "nominal", "label": "Cost scenario"}, {"field": "cagr_net", "type": "quantitative", "label": "Net CAGR", "format": "percent"}, value_format="percent", intent="comparison", tooltip=[{"field": "cumulative_return_net", "type": "quantitative", "label": "Cumulative return", "format": "percent"}, {"field": "cost_drag", "type": "quantitative", "label": "Cost drag", "format": "percent"}]),
        _chart("benchmark_cagr", "CAGR ของ FRTBOT Base และ benchmarks", "ช่วง OOS เดียวกัน; proxy returns แปลงเป็น THB", "horizontalBar", "benchmark_comparison", "performance_export", {"field": "display_name", "type": "nominal", "label": "Portfolio"}, {"field": "cagr_net", "type": "quantitative", "label": "Net CAGR", "format": "percent"}, value_format="percent", intent="comparison", tooltip=[{"field": "sharpe_net", "type": "quantitative", "label": "Sharpe"}, {"field": "max_drawdown_net", "type": "quantitative", "label": "Max drawdown", "format": "percent"}]),
        _chart("fold_comparison", "ผลตอบแทนสะสมราย walk-forward fold", "FRTBOT Base เทียบ equal-weight บน test window เดียวกัน", "bar", "fold_comparison", "fold_export", {"field": "fold_label", "type": "ordinal", "label": "Test fold"}, {"field": "cumulative_return", "type": "quantitative", "label": "Cumulative return", "format": "percent"}, color={"field": "series", "type": "nominal", "label": "Series"}, value_format="percent", intent="comparison"),
        _chart("model_rank_ic", "Rank IC เฉลี่ยของโมเดล", "ค่าเฉลี่ยข้าม 5 test folds; สูงกว่าศูนย์หมายถึงอันดับไปทิศเดียวกับผลตอบแทนจริง", "horizontalBar", "model_average", "model_export", {"field": "display_name", "type": "nominal", "label": "Model"}, {"field": "rank_ic", "type": "quantitative", "label": "Mean Rank IC"}, value_format="number", intent="comparison", tooltip=[{"field": "top_quintile_hit_rate", "type": "quantitative", "label": "Top-quintile hit rate", "format": "percent"}, {"field": "rank_ic_std", "type": "quantitative", "label": "Rank IC std"}]),
        _chart("allocation_mix", "องค์ประกอบน้ำหนักพอร์ตรายวัน", "น้ำหนัก implemented หลัง next-bar shift; รวม CASH_THB", "stackedArea", "allocation_daily", "allocation_export", {"field": "date", "type": "temporal", "label": "Date"}, {"field": "weight", "type": "quantitative", "label": "Portfolio weight", "format": "percent"}, color={"field": "asset", "type": "nominal", "label": "Asset"}, value_format="percent", intent="composition"),
        _chart("data_coverage", "Business-day coverage ของ market และ FX series", "ข้อมูล local cache ถึง 30 ก.ค. 2026; วันหยุดตลาดทำให้ market proxy ต่ำกว่า FX", "bar", "audit", "data_audit_export", {"field": "key", "type": "nominal", "label": "Series"}, {"field": "coverage", "type": "quantitative", "label": "Coverage ratio", "format": "percent"}, value_format="percent", intent="comparison", tooltip=[{"field": "row_count", "type": "quantitative", "label": "Rows"}, {"field": "max_gap_days", "type": "quantitative", "label": "Max gap days"}, {"field": "anomaly_count", "type": "quantitative", "label": "Anomalies"}]),
    ]

    scenario_columns = [
        {"field": "display_name", "label": "Scenario", "type": "text"},
        {"field": "cagr_net", "label": "CAGR", "format": "percent"},
        {"field": "cumulative_return_net", "label": "Cumulative", "format": "percent"},
        {"field": "annualized_vol_net", "label": "Ann. vol", "format": "percent"},
        {"field": "sharpe_net", "label": "Sharpe", "format": "number"},
        {"field": "sortino_net", "label": "Sortino", "format": "number"},
        {"field": "max_drawdown_net", "label": "Max DD", "format": "percent"},
        {"field": "cost_drag", "label": "Cost drag", "format": "percent"},
        {"field": "total_turnover", "label": "Total turnover", "format": "number"},
    ]
    tables = [
        {"id": "scenario_detail", "title": "รายละเอียดผลตอบแทนตามต้นทุน", "subtitle": "ค่ารวมทุก OOS fold, 1,084 trading days", "dataset": "scenarios", "sourceId": "performance_export", "defaultSort": {"field": "cagr_net", "direction": "desc"}, "columns": scenario_columns},
        {"id": "benchmark_detail", "title": "รายละเอียด strategy และ benchmark", "subtitle": "Buy-and-hold และ equal-weight ใช้ fold-reset convention เดียวกับ strategy เพื่อเทียบกันได้", "dataset": "benchmark_comparison", "sourceId": "performance_export", "defaultSort": {"field": "cagr_net", "direction": "desc"}, "columns": [{"field": "display_name", "label": "Portfolio", "type": "text"}, {"field": "cagr_net", "label": "CAGR", "format": "percent"}, {"field": "cumulative_return_net", "label": "Cumulative", "format": "percent"}, {"field": "annualized_vol_net", "label": "Ann. vol", "format": "percent"}, {"field": "sharpe_net", "label": "Sharpe", "format": "number"}, {"field": "sortino_net", "label": "Sortino", "format": "number"}, {"field": "max_drawdown_net", "label": "Max DD", "format": "percent"}, {"field": "cost_drag", "label": "Cost drag", "format": "percent"}]},
        {"id": "fold_detail", "title": "Walk-forward fold diagnostics", "subtitle": "Fold 4 เป็น partial year 44 trading days", "dataset": "folds", "sourceId": "fold_export", "defaultSort": {"field": "fold", "direction": "asc"}, "columns": [{"field": "fold", "label": "Fold", "type": "number"}, {"field": "test_start", "label": "Test start", "type": "date"}, {"field": "test_end", "label": "Test end", "type": "date"}, {"field": "cumulative_return_net", "label": "FRTBOT cumulative", "format": "percent"}, {"field": "equal_weight_cumulative_return_net", "label": "Equal-weight cumulative", "format": "percent"}, {"field": "cagr_spread_vs_equal_weight", "label": "CAGR spread", "format": "percent", "movement": True}, {"field": "max_drawdown_net", "label": "FRTBOT Max DD", "format": "percent"}, {"field": "test_rows", "label": "Test rows", "format": "number"}]},
        {"id": "model_detail", "title": "Model metrics เฉลี่ย", "subtitle": "ค่าเฉลี่ยและ dispersion ข้าม 5 folds; equal-score Rank IC เท่ากับศูนย์โดยโครงสร้าง", "dataset": "model_average", "sourceId": "model_export", "defaultSort": {"field": "rank_ic", "direction": "desc"}, "columns": [{"field": "display_name", "label": "Model", "type": "text"}, {"field": "rank_ic", "label": "Mean Rank IC", "format": "number"}, {"field": "rank_ic_std", "label": "IC std", "format": "number"}, {"field": "top_quintile_hit_rate", "label": "Top-quintile hit", "format": "percent"}, {"field": "scored_rows", "label": "Scored rows", "format": "number"}]},
        {"id": "allocation_detail", "title": "Allocation summary และ contribution", "subtitle": "Contribution เป็น arithmetic gross approximation; cash ไม่มี country contribution", "dataset": "allocation_summary", "sourceId": "allocation_export", "defaultSort": {"field": "average_weight", "direction": "desc"}, "columns": [{"field": "asset", "label": "Asset", "type": "text"}, {"field": "average_weight", "label": "Avg weight", "format": "percent"}, {"field": "maximum_weight", "label": "Max weight", "format": "percent"}, {"field": "days_nonzero_share", "label": "Days held", "format": "percent"}, {"field": "gross_return_contribution", "label": "Gross contribution", "format": "percent"}]},
        {"id": "data_audit", "title": "Data-quality audit", "subtitle": "9 real local series; JP anomaly dates trigger modeling exclusion", "dataset": "audit", "sourceId": "data_audit_export", "defaultSort": {"field": "coverage", "direction": "asc"}, "columns": [{"field": "key", "label": "Series", "type": "text"}, {"field": "kind", "label": "Kind", "type": "text"}, {"field": "provider_symbol", "label": "Symbol", "type": "text"}, {"field": "row_count", "label": "Rows", "format": "number"}, {"field": "coverage", "label": "Coverage", "format": "percent"}, {"field": "max_gap_days", "label": "Max gap days", "format": "number"}, {"field": "anomaly_count", "label": "Anomalies", "format": "number"}, {"field": "last_valid_date", "label": "Last date", "type": "date"}]},
        {"id": "metric_definitions", "title": "Finance metrics แปลเป็นภาษาคน", "subtitle": "สูตรและความหมายที่ใช้ตีความตัวเลขในรายงาน", "dataset": "metric_definitions", "sourceId": "method_export", "defaultSort": {"field": "metric", "direction": "asc"}, "columns": [{"field": "metric", "label": "Metric", "type": "text"}, {"field": "finance_language", "label": "ภาษาการเงิน", "type": "text"}, {"field": "plain_thai", "label": "ขยายความ", "type": "text"}]},
        {"id": "engineering_components", "title": "Engineering component map", "subtitle": "Reusable logic อยู่ใน src/frtbot; notebooks เป็น orchestration/reporting", "dataset": "engineering", "sourceId": "method_export", "defaultSort": {"field": "layer", "direction": "asc"}, "columns": [{"field": "layer", "label": "Layer", "type": "text"}, {"field": "module", "label": "Module", "type": "text"}, {"field": "role", "label": "หน้าที่", "type": "text"}]},
        {"id": "dataset_inventory", "title": "Local evidence inventory", "subtitle": "ทุกไฟล์อยู่ใต้ artifacts/detailed_report/data และไม่ถูกอัปโหลด", "dataset": "inventory", "sourceId": "inventory_export", "defaultSort": {"field": "file", "direction": "asc"}, "columns": [{"field": "file", "label": "File", "type": "text"}, {"field": "rows", "label": "Rows", "format": "number"}, {"field": "size_bytes", "label": "Bytes", "format": "compact"}, {"field": "description", "label": "Contents", "type": "text"}]},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FRTBOT: รายงาน Backtest เชิงการเงินและวิศวกรรม",
            "description": "Technical, source-backed review of the five-fold THB walk-forward research backtest.",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline": _records(headline),
                "equity_monthly": _records(equity_monthly),
                "scenarios": _records(scenarios),
                "benchmark_comparison": _records(pd.concat([scenarios.loc[scenarios["series"] == "base"], benchmarks], ignore_index=True, sort=False)),
                "fold_comparison": _records(fold_comparison),
                "folds": _records(folds),
                "model_average": _records(model_average),
                "allocation_daily": _records(allocation_daily),
                "allocation_summary": _records(allocation_summary),
                "audit": _records(audit),
                "anomalies": _records(anomalies),
                "metric_definitions": _records(metric_definitions),
                "engineering": _records(engineering),
                "inventory": _records(dataset_inventory_frame),
                "feature_catalog": _records(feature_catalog),
                "rebalance_weights": _records(rebalance_weights),
                "model_folds": _records(model_folds),
            },
        },
        "sources": sources,
        "package_info": {
            "root": "artifacts/detailed_report",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json",
            "localOnly": True,
        },
    }

    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Canonical artifact written to {output}")
    print(f"Snapshot rows: {sum(len(rows) for rows in artifact['snapshot']['datasets'].values())}")


if __name__ == "__main__":
    main()
