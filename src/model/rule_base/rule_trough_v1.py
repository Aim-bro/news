#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-Based Trough Detection Strategy v1

ML Feature Importance 기반으로 룰 전략 구성:
  1위. volatility_20     → 변동성이 높을 때
  2위. index_ret_20      → 시장이 하락했을 때
  3위. macd_hist          → MACD 히스토그램 바닥
  4위. 기관_비중           → 기관 참여도
  5위. index_ret_5        → 단기 시장 하락

Phase 1: 피처 분포 분석 (threshold 결정용)
Phase 2: 룰 기반 저점 탐지
Phase 3: Walk-Forward 백테스트

Input:  data/features_lgbm.csv  (preprocess_lgbm.py 출력물)
Output: rule_analysis.json, rule_trades.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

HORIZON = 10       # target: 향후 10일 내 10% 상승
THRESHOLD = 0.10

# preprocess_lgbm.py와 동일한 헬퍼
def rolling_zscore(series, window=20, min_periods=10):
    rm = series.rolling(window, min_periods=min_periods).mean()
    rs = series.rolling(window, min_periods=min_periods).std()
    return (series - rm) / rs.replace(0, np.nan)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "날짜" in df.columns:
        df["date"] = pd.to_datetime(df["날짜"], errors="coerce")

    if "stock_code" in df.columns:
        code_src = df["stock_code"]
    elif "종목코드" in df.columns:
        code_src = df["종목코드"]
    else:
        raise ValueError("No stock code column")

    df["stock_code"] = code_src.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return df


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features needed for rule strategy (if not already present)."""
    g = df.groupby("stock_code", sort=False)

    # volatility_20
    if "volatility_20" not in df.columns:
        df["ret_1"] = g["종가"].pct_change(1)
        df["volatility_20"] = g["ret_1"].transform(
            lambda x: x.rolling(20, min_periods=15).std()
        )

    # RSI_14
    if "RSI_14" not in df.columns:
        df["RSI_14"] = g.apply(lambda x: calc_rsi(x["종가"], 14)).reset_index(level=0, drop=True)

    # MACD histogram
    if "macd_hist" not in df.columns:
        macd_results = g.apply(lambda x: pd.DataFrame(
            dict(zip(["_macd", "_signal", "_hist"], calc_macd(x["종가"]))),
            index=x.index,
        ))
        if isinstance(macd_results.index, pd.MultiIndex):
            macd_results = macd_results.droplevel(0)
        df["macd_hist"] = macd_results["_hist"]

    # MACD hist diff (변화 방향)
    if "hist_diff" not in df.columns:
        df["hist_diff"] = g["macd_hist"].diff(1)

    # disparity_20
    if "disparity_20" not in df.columns:
        ma20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).mean())
        df["disparity_20"] = df["종가"] / ma20 * 100

    # BB %B
    if "BB_pctB" not in df.columns:
        ma20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).mean())
        std20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).std())
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        df["BB_pctB"] = (df["종가"] - bb_lower) / bb_width

    return df


def add_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """Add trough target: 향후 N일 내 최대 상승 >= threshold."""
    g = df.groupby("stock_code", sort=False)
    future_max = g["종가"].transform(
        lambda x: x.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    max_rally = future_max / df["종가"] - 1.0
    df["target_trough"] = (max_rally >= threshold).astype(float)
    df.loc[future_max.isna(), "target_trough"] = np.nan
    df["fwd_max_rally"] = max_rally
    return df


# ─── Phase 1: Feature Distribution Analysis ─────────────────────────────────

def analyze_distributions(df: pd.DataFrame):
    """Analyze feature distributions to determine rule thresholds."""
    print("=" * 60)
    print("Phase 1: Feature Distribution Analysis")
    print("=" * 60)

    # Trough인 날 vs 아닌 날의 피처 분포 비교
    valid = df[df["target_trough"].notna()].copy()
    trough = valid[valid["target_trough"] == 1]
    normal = valid[valid["target_trough"] == 0]

    print(f"\nTotal: {len(valid):,} rows")
    print(f"Trough (저점): {len(trough):,} ({len(trough)/len(valid)*100:.1f}%)")
    print(f"Normal: {len(normal):,} ({len(normal)/len(valid)*100:.1f}%)")

    features_to_analyze = [
        "volatility_20", "index_ret_20", "index_ret_5", "index_ret_1",
        "RSI_14", "macd_hist", "hist_diff", "disparity_20", "BB_pctB",
        "ret_20", "ret_5",
    ]

    print(f"\n{'Feature':<20s} {'Normal_med':>12s} {'Trough_med':>12s} {'Diff':>10s} {'Direction':>10s}")
    print("-" * 70)

    insights = {}
    for feat in features_to_analyze:
        if feat not in df.columns:
            continue
        n_med = normal[feat].median()
        t_med = trough[feat].median()
        diff = t_med - n_med

        if abs(n_med) > 0.001:
            direction = "↑ higher" if diff > 0 else "↓ lower"
        else:
            direction = "↑" if diff > 0 else "↓"

        print(f"{feat:<20s} {n_med:>12.4f} {t_med:>12.4f} {diff:>10.4f} {direction:>10s}")
        insights[feat] = {
            "normal_median": float(n_med),
            "trough_median": float(t_med),
            "trough_p25": float(trough[feat].quantile(0.25)),
            "trough_p75": float(trough[feat].quantile(0.75)),
            "normal_p75": float(normal[feat].quantile(0.75)),
            "normal_p25": float(normal[feat].quantile(0.25)),
        }

    # Percentile analysis for key features
    print(f"\n{'─'*60}")
    print("Key Feature Percentiles (Trough days)")
    print(f"{'─'*60}")

    for feat in ["volatility_20", "RSI_14", "index_ret_20", "disparity_20", "BB_pctB"]:
        if feat not in df.columns:
            continue
        print(f"\n  {feat}:")
        for pct in [10, 25, 50, 75, 90]:
            n_val = normal[feat].quantile(pct/100)
            t_val = trough[feat].quantile(pct/100)
            print(f"    P{pct:02d}: Normal={n_val:>8.4f}  Trough={t_val:>8.4f}")

    return insights


# ─── Phase 2: Rule-Based Strategy ────────────────────────────────────────────

class TroughRuleStrategy:
    """
    Rule-based trough detection using ML-discovered features.

    Signal conditions (AND logic):
      Base: volatility_20 > vol_threshold (시장 공포)
      + at least min_conditions of:
        - RSI_14 < rsi_threshold (과매도)
        - index_ret_20 < idx_threshold (시장 하락)
        - disparity_20 < disp_threshold (이평선 하회)
        - BB_pctB < bb_threshold (볼린저 하단)
        - hist_diff > 0 (MACD 반전 시작)
    """

    def __init__(self,
                 vol_threshold: float = None,      # volatility_20 상위 percentile
                 rsi_threshold: float = 40.0,
                 idx_ret_20_threshold: float = -0.05,
                 disp_threshold: float = 97.0,
                 bb_threshold: float = 0.2,
                 hist_diff_positive: bool = True,
                 min_conditions: int = 2,
                 name: str = "default"):
        self.vol_threshold = vol_threshold
        self.rsi_threshold = rsi_threshold
        self.idx_ret_20_threshold = idx_ret_20_threshold
        self.disp_threshold = disp_threshold
        self.bb_threshold = bb_threshold
        self.hist_diff_positive = hist_diff_positive
        self.min_conditions = min_conditions
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate binary buy signals."""

        # Base condition: high volatility (공포 상태)
        if self.vol_threshold is not None:
            base = df["volatility_20"] > self.vol_threshold
        else:
            # Auto: 종목별 상위 30%
            vol_rank = df.groupby("stock_code")["volatility_20"].rank(pct=True)
            base = vol_rank > 0.70

        # Sub-conditions (count how many are met)
        conditions = pd.DataFrame(index=df.index)

        if "RSI_14" in df.columns:
            conditions["rsi"] = (df["RSI_14"] < self.rsi_threshold).astype(int)

        if "index_ret_20" in df.columns:
            conditions["idx_ret"] = (df["index_ret_20"] < self.idx_ret_20_threshold).astype(int)

        if "disparity_20" in df.columns:
            conditions["disp"] = (df["disparity_20"] < self.disp_threshold).astype(int)

        if "BB_pctB" in df.columns:
            conditions["bb"] = (df["BB_pctB"] < self.bb_threshold).astype(int)

        if "hist_diff" in df.columns and self.hist_diff_positive:
            conditions["macd_turn"] = (df["hist_diff"] > 0).astype(int)

        met_count = conditions.sum(axis=1)
        signal = base & (met_count >= self.min_conditions)

        return signal.astype(int)

    def describe(self) -> str:
        lines = [f"Strategy: {self.name}"]
        lines.append(f"  Base: volatility_20 > {self.vol_threshold or 'top 30%'}")
        lines.append(f"  + at least {self.min_conditions} of:")
        lines.append(f"    - RSI_14 < {self.rsi_threshold}")
        lines.append(f"    - index_ret_20 < {self.idx_ret_20_threshold}")
        lines.append(f"    - disparity_20 < {self.disp_threshold}")
        lines.append(f"    - BB_pctB < {self.bb_threshold}")
        if self.hist_diff_positive:
            lines.append(f"    - hist_diff > 0 (MACD turning up)")
        return "\n".join(lines)


# ─── Phase 3: Walk-Forward Backtest ──────────────────────────────────────────

def walk_forward_backtest(df: pd.DataFrame, strategy: TroughRuleStrategy,
                          initial_train_end: str = "2024-06-30",
                          test_window_days: int = 60,
                          max_holding: int = 20,
                          stop_loss: float = -0.15) -> Dict:
    """
    Walk-forward backtest for rule strategy.
    Uses train period only for distribution analysis (no fitting).
    """
    dates = np.array(sorted(df["date"].dropna().unique()))
    init_end = pd.Timestamp(initial_train_end)
    train_end_idx = np.searchsorted(dates, init_end.to_datetime64(), side="right") - 1
    train_end_idx = min(train_end_idx, len(dates) - 2)

    all_trades = []
    all_metrics = []
    fold_id = 0

    while True:
        test_start_idx = train_end_idx + 1
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_start_idx + test_window_days - 1, len(dates) - 1)
        fold_id += 1

        test_start = pd.Timestamp(dates[test_start_idx])
        test_end = pd.Timestamp(dates[test_end_idx])
        train_end = pd.Timestamp(dates[train_end_idx])

        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if len(test_data) == 0:
            break

        # Generate signals
        signals = strategy.generate_signals(test_data)
        test_data = test_data.copy()
        test_data["signal"] = signals.values

        # Evaluate signal quality (if target available)
        if "target_trough" in test_data.columns:
            valid_mask = test_data["target_trough"].notna() & test_data["signal"].notna()
            valid = test_data[valid_mask]
            if len(valid) > 0:
                sig_on = valid[valid["signal"] == 1]
                sig_off = valid[valid["signal"] == 0]

                total_signals = len(sig_on)
                if total_signals > 0:
                    precision = float(sig_on["target_trough"].mean())
                else:
                    precision = 0.0

                total_troughs = int(valid["target_trough"].sum())
                if total_troughs > 0:
                    recall = float(sig_on["target_trough"].sum() / total_troughs)
                else:
                    recall = 0.0

                all_metrics.append({
                    "fold": fold_id,
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "test_rows": len(valid),
                    "total_signals": total_signals,
                    "signal_rate": float(signals.mean()),
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                    "trough_rate": float(valid["target_trough"].mean()),
                })

        # Simulate trades
        for stock_code, g_df in test_data.groupby("stock_code", sort=False):
            pos = None
            for _, row in g_df.iterrows():
                px = float(row["종가"])
                sig = int(row["signal"])
                dt = pd.Timestamp(row["date"])

                if pos is None:
                    if sig == 1:
                        pos = {"stock_code": stock_code, "entry_date": dt,
                               "entry_price": px, "bars": 0}
                    continue

                pos["bars"] += 1
                ret_now = px / pos["entry_price"] - 1.0

                exit_by_time = pos["bars"] >= max_holding
                exit_by_stop = ret_now <= stop_loss
                exit_by_profit = ret_now >= THRESHOLD  # take profit at target

                if exit_by_time or exit_by_stop or exit_by_profit:
                    all_trades.append({
                        "fold": fold_id,
                        "stock_code": pos["stock_code"],
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": dt.strftime("%Y-%m-%d"),
                        "entry_price": pos["entry_price"],
                        "exit_price": px,
                        "return": ret_now,
                        "holding_days": pos["bars"],
                        "exit_reason": "profit" if exit_by_profit else ("stop" if exit_by_stop else "time"),
                    })
                    pos = None

        train_end_idx = test_end_idx
        if train_end_idx >= len(dates) - 2:
            break

    # Summarize
    if all_trades:
        tdf = pd.DataFrame(all_trades)
        equity = (1.0 + tdf["return"].astype(float)).cumprod()
        dd = equity / equity.cummax() - 1.0
        wins = tdf[tdf["return"] > 0]["return"]
        losses = tdf[tdf["return"] <= 0]["return"]
        wl = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan

        trade_summary = {
            "total_trades": len(tdf),
            "win_rate": float((tdf["return"] > 0).mean()),
            "cumulative_return": float(equity.iloc[-1] - 1.0),
            "avg_return": float(tdf["return"].mean()),
            "wl_ratio": wl,
            "avg_holding_days": float(tdf["holding_days"].mean()),
            "max_drawdown": float(dd.min()),
            "profit_exits": int((tdf["exit_reason"] == "profit").sum()),
            "stop_exits": int((tdf["exit_reason"] == "stop").sum()),
            "time_exits": int((tdf["exit_reason"] == "time").sum()),
        }
    else:
        trade_summary = {"total_trades": 0}

    return {
        "strategy": strategy.name,
        "folds": len(all_metrics),
        "fold_metrics": all_metrics,
        "trade_summary": trade_summary,
        "trades": all_trades,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="Rule-based Trough Detection")
    ap.add_argument("--data", default="data/features_lgbm.csv")
    ap.add_argument("--out-analysis", default="rule_analysis.json")
    ap.add_argument("--out-trades", default="rule_trades.json")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    print("=" * 60)
    print("Rule-Based Trough Detection Strategy")
    print(f"  Target: 향후 {HORIZON}일 내 {THRESHOLD*100:.0f}%+ 상승")
    print("=" * 60)

    # Load & prepare
    df = load_data(csv_path)
    df = add_rule_features(df)
    df = add_target(df, HORIZON, THRESHOLD)
    print(f"Data: {len(df):,} rows, {df['stock_code'].nunique()} stocks")

    trough_rate = df["target_trough"].dropna().mean()
    print(f"Trough rate: {trough_rate*100:.1f}%")

    # Phase 1: Distribution analysis
    insights = analyze_distributions(df)

    # Phase 2: Define strategies
    strategies = [
        # Conservative: 엄격한 조건 (높은 Precision, 낮은 Recall)
        TroughRuleStrategy(
            vol_threshold=None,  # auto top 30%
            rsi_threshold=35,
            idx_ret_20_threshold=-0.05,
            disp_threshold=96,
            bb_threshold=0.15,
            hist_diff_positive=True,
            min_conditions=3,
            name="conservative",
        ),
        # Balanced: 중간 조건
        TroughRuleStrategy(
            vol_threshold=None,
            rsi_threshold=40,
            idx_ret_20_threshold=-0.03,
            disp_threshold=98,
            bb_threshold=0.25,
            hist_diff_positive=True,
            min_conditions=2,
            name="balanced",
        ),
        # Aggressive: 느슨한 조건 (낮은 Precision, 높은 Recall)
        TroughRuleStrategy(
            vol_threshold=None,
            rsi_threshold=45,
            idx_ret_20_threshold=-0.02,
            disp_threshold=99,
            bb_threshold=0.3,
            hist_diff_positive=True,
            min_conditions=2,
            name="aggressive",
        ),
    ]

    # Phase 3: Walk-forward backtest
    print(f"\n{'='*60}")
    print("Phase 3: Walk-Forward Backtest")
    print(f"{'='*60}")

    all_results = []
    for strat in strategies:
        print(f"\n{'─'*50}")
        print(strat.describe())
        print(f"{'─'*50}")

        result = walk_forward_backtest(df, strat)
        all_results.append(result)

        # Print fold metrics
        if result["fold_metrics"]:
            mdf = pd.DataFrame(result["fold_metrics"])
            print(f"\n  Folds: {len(mdf)}")
            print(f"  Signal Rate:  {mdf['signal_rate'].mean():.4f} ± {mdf['signal_rate'].std():.4f}")
            print(f"  Precision:    {mdf['precision'].mean():.4f} ± {mdf['precision'].std():.4f}")
            print(f"  Recall:       {mdf['recall'].mean():.4f} ± {mdf['recall'].std():.4f}")
            print(f"  F1:           {mdf['f1'].mean():.4f} ± {mdf['f1'].std():.4f}")

        ts = result["trade_summary"]
        if ts.get("total_trades", 0) > 0:
            print(f"\n  Trades: {ts['total_trades']}")
            print(f"  Win Rate:     {ts['win_rate']:.4f}")
            print(f"  Cum Return:   {ts['cumulative_return']:.4f}")
            print(f"  Avg Return:   {ts['avg_return']:.4f}")
            print(f"  W/L Ratio:    {ts['wl_ratio']:.4f}")
            print(f"  Avg Holding:  {ts['avg_holding_days']:.1f} days")
            print(f"  Max Drawdown: {ts['max_drawdown']:.4f}")
            print(f"  Exits: profit={ts['profit_exits']}, stop={ts['stop_exits']}, time={ts['time_exits']}")
        else:
            print("\n  No trades generated.")

    # Summary comparison
    print(f"\n{'='*60}")
    print("Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<15s} {'Signals':>8s} {'Trades':>7s} {'WinRate':>8s} {'Precision':>10s} "
          f"{'Recall':>8s} {'F1':>8s} {'CumRet':>10s} {'MDD':>8s}")
    print("-" * 95)

    for r in all_results:
        ts = r["trade_summary"]
        if r["fold_metrics"]:
            mdf = pd.DataFrame(r["fold_metrics"])
            avg_sig = mdf["total_signals"].sum()
            avg_prec = mdf["precision"].mean()
            avg_rec = mdf["recall"].mean()
            avg_f1 = mdf["f1"].mean()
        else:
            avg_sig = avg_prec = avg_rec = avg_f1 = 0

        trades = ts.get("total_trades", 0)
        wr = ts.get("win_rate", 0)
        cr = ts.get("cumulative_return", 0)
        mdd = ts.get("max_drawdown", 0)

        print(f"{r['strategy']:<15s} {avg_sig:>8.0f} {trades:>7d} {wr:>8.4f} {avg_prec:>10.4f} "
              f"{avg_rec:>8.4f} {avg_f1:>8.4f} {cr:>10.4f} {mdd:>8.4f}")

    # vs ML comparison
    print(f"\n{'='*60}")
    print("vs ML Model (v6 Trough H10)")
    print(f"{'='*60}")
    print(f"  ML AUC:       0.5857")
    print(f"  ML Precision: 0.3171")
    print(f"  ML Recall:    0.4170")
    print(f"  ML F1:        0.3520")
    print(f"  (Rule strategies above for comparison)")

    # Save
    output = {
        "config": {
            "horizon": HORIZON, "threshold": THRESHOLD,
            "data": str(csv_path),
        },
        "distribution_insights": insights,
        "strategies": [
            {
                "name": r["strategy"],
                "fold_metrics": r["fold_metrics"],
                "trade_summary": r["trade_summary"],
            }
            for r in all_results
        ],
    }
    Path(args.out_analysis).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    all_trades = []
    for r in all_results:
        for t in r["trades"]:
            t["strategy"] = r["strategy"]
            all_trades.append(t)
    Path(args.out_trades).write_text(
        json.dumps(all_trades, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved: {args.out_analysis}, {args.out_trades}")
    print(f"[TIME] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
