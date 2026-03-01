#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-Based Trough Detection v2 — "진짜 저점 반등"

기존 v1 문제: "10일 내 10% 상승" = 모멘텀 가속 (이미 오르는 종목)
v2 해결: "최근 크게 하락한 종목이 반등하는가?" = 진짜 저점 반등

타겟 재정의:
  조건1 (하락 필터): 최근 20일 수익률 <= -X% (충분히 하락한 상태)
  조건2 (반등 타겟): 향후 10일 내 최대 상승 >= Y%

조합 비교:
  A: ret_20 <= -10%, 향후 10일 내 5%+ 반등
  B: ret_20 <= -15%, 향후 10일 내 5%+ 반등
  C: ret_20 <= -10%, 향후 10일 내 7%+ 반등

Input:  data/features_lgbm.csv
Output: rule_v2_analysis.json, rule_v2_trades.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all needed features exist."""
    g = df.groupby("stock_code", sort=False)

    if "ret_1" not in df.columns:
        df["ret_1"] = g["종가"].pct_change(1)
    if "ret_5" not in df.columns:
        df["ret_5"] = g["종가"].pct_change(5)
    if "ret_20" not in df.columns:
        df["ret_20"] = g["종가"].pct_change(20)

    if "volatility_20" not in df.columns:
        df["volatility_20"] = g["ret_1"].transform(
            lambda x: x.rolling(20, min_periods=15).std()
        )

    if "RSI_14" not in df.columns:
        rsi_vals = g["종가"].transform(lambda x: calc_rsi(x, 14))
        df["RSI_14"] = rsi_vals

    if "macd_hist" not in df.columns:
        all_hist = []
        for _, grp in g:
            _, _, hist = calc_macd(grp["종가"])
            all_hist.append(hist)
        df["macd_hist"] = pd.concat(all_hist).values

    if "hist_diff" not in df.columns:
        df["hist_diff"] = g["macd_hist"].diff(1)

    if "disparity_20" not in df.columns:
        ma20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).mean())
        df["disparity_20"] = df["종가"] / ma20 * 100

    if "BB_pctB" not in df.columns:
        ma20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).mean())
        std20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).std())
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        df["BB_pctB"] = (df["종가"] - bb_lower) / bb_width

    return df


def add_target_v2(df: pd.DataFrame, drop_threshold: float,
                  bounce_horizon: int, bounce_threshold: float) -> pd.DataFrame:
    """
    True trough target:
      1) ret_20 <= -drop_threshold (최근 20일 충분히 하락)
      2) 향후 bounce_horizon일 내 bounce_threshold 이상 반등 → 1

    Only rows meeting condition 1 are eligible.
    Rows not meeting condition 1 get target = NaN (excluded from evaluation).
    """
    g = df.groupby("stock_code", sort=False)

    # Forward max price
    future_max = g["종가"].transform(
        lambda x: x.iloc[::-1].rolling(bounce_horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    max_rally = future_max / df["종가"] - 1.0

    # Drop filter: 최근 20일 수익률
    is_dropped = df["ret_20"] <= -drop_threshold

    # Target: dropped AND bounced
    target = pd.Series(np.nan, index=df.index)
    eligible = is_dropped & future_max.notna()
    target[eligible] = (max_rally[eligible] >= bounce_threshold).astype(float)

    df["is_dropped"] = is_dropped.astype(int)
    df["target_trough_v2"] = target
    df["fwd_max_rally"] = max_rally

    return df


# ─── Distribution Analysis ───────────────────────────────────────────────────

def analyze_trough_distribution(df: pd.DataFrame, label: str):
    """Analyze features of true trough vs false trough among dropped stocks."""
    valid = df[df["target_trough_v2"].notna()].copy()
    if len(valid) == 0:
        print(f"  [{label}] No eligible rows.")
        return {}

    trough = valid[valid["target_trough_v2"] == 1]
    no_bounce = valid[valid["target_trough_v2"] == 0]

    print(f"\n  [{label}] Eligible (dropped): {len(valid):,}")
    print(f"    Bounced (target=1): {len(trough):,} ({len(trough)/len(valid)*100:.1f}%)")
    print(f"    No bounce (target=0): {len(no_bounce):,} ({len(no_bounce)/len(valid)*100:.1f}%)")

    features = ["RSI_14", "volatility_20", "disparity_20", "BB_pctB",
                "macd_hist", "hist_diff", "ret_5"]

    insights = {}
    print(f"\n    {'Feature':<18s} {'NoBounce_med':>14s} {'Bounced_med':>14s} {'Direction':>10s}")
    print(f"    {'-'*60}")

    for feat in features:
        if feat not in df.columns:
            continue
        nb_med = no_bounce[feat].median()
        b_med = trough[feat].median()
        diff = b_med - nb_med
        direction = "↑ higher" if diff > 0 else "↓ lower"
        print(f"    {feat:<18s} {nb_med:>14.4f} {b_med:>14.4f} {direction:>10s}")
        insights[feat] = {"no_bounce_median": float(nb_med), "bounced_median": float(b_med)}

    return insights


# ─── Rule Strategy ───────────────────────────────────────────────────────────

class TroughRuleV2:
    """
    Rule for detecting bounce from true trough.

    Prerequisite: ret_20 <= -drop_threshold (already dropped)
    Buy signal: at least min_conditions of:
      - RSI_14 < rsi_max (still oversold)
      - BB_pctB < bb_max (near lower band)
      - disparity_20 < disp_max (below MA)
      - hist_diff > 0 (MACD turning up)
      - volatility_20 > vol_min (fear/volatility present)
    """

    def __init__(self,
                 drop_threshold: float = 0.10,
                 rsi_max: float = 40,
                 bb_max: float = 0.25,
                 disp_max: float = 97,
                 hist_diff_positive: bool = True,
                 vol_percentile: float = 0.60,
                 min_conditions: int = 2,
                 name: str = "default"):
        self.drop_threshold = drop_threshold
        self.rsi_max = rsi_max
        self.bb_max = bb_max
        self.disp_max = disp_max
        self.hist_diff_positive = hist_diff_positive
        self.vol_percentile = vol_percentile
        self.min_conditions = min_conditions
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Prerequisite: stock has dropped
        is_dropped = df["ret_20"] <= -self.drop_threshold

        # Sub-conditions
        conditions = pd.DataFrame(index=df.index)

        if "RSI_14" in df.columns:
            conditions["rsi"] = (df["RSI_14"] < self.rsi_max).astype(int)

        if "BB_pctB" in df.columns:
            conditions["bb"] = (df["BB_pctB"] < self.bb_max).astype(int)

        if "disparity_20" in df.columns:
            conditions["disp"] = (df["disparity_20"] < self.disp_max).astype(int)

        if "hist_diff" in df.columns and self.hist_diff_positive:
            conditions["macd_turn"] = (df["hist_diff"] > 0).astype(int)

        if "volatility_20" in df.columns:
            vol_rank = df.groupby("stock_code")["volatility_20"].rank(pct=True)
            conditions["vol"] = (vol_rank > self.vol_percentile).astype(int)

        met_count = conditions.sum(axis=1)
        signal = is_dropped & (met_count >= self.min_conditions)

        return signal.astype(int)

    def describe(self) -> str:
        lines = [f"Strategy: {self.name}"]
        lines.append(f"  Prerequisite: ret_20 <= -{self.drop_threshold*100:.0f}%")
        lines.append(f"  + at least {self.min_conditions} of:")
        lines.append(f"    - RSI_14 < {self.rsi_max}")
        lines.append(f"    - BB_pctB < {self.bb_max}")
        lines.append(f"    - disparity_20 < {self.disp_max}")
        if self.hist_diff_positive:
            lines.append(f"    - hist_diff > 0 (MACD turning up)")
        lines.append(f"    - volatility_20 > P{self.vol_percentile*100:.0f}")
        return "\n".join(lines)


# ─── Backtest ────────────────────────────────────────────────────────────────

def walk_forward_backtest(df, strategy, initial_train_end="2024-06-30",
                          test_window_days=60, max_holding=20, stop_loss=-0.15,
                          take_profit=0.05):
    dates = np.array(sorted(df["date"].dropna().unique()))
    init_end = pd.Timestamp(initial_train_end)
    train_end_idx = np.searchsorted(dates, init_end.to_datetime64(), side="right") - 1
    train_end_idx = min(train_end_idx, len(dates) - 2)

    all_trades, all_metrics = [], []
    fold_id = 0

    while True:
        test_start_idx = train_end_idx + 1
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_start_idx + test_window_days - 1, len(dates) - 1)
        fold_id += 1

        test_start = pd.Timestamp(dates[test_start_idx])
        test_end = pd.Timestamp(dates[test_end_idx])

        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if len(test_data) == 0:
            break

        signals = strategy.generate_signals(test_data)
        test_data["signal"] = signals.values

        # Evaluate (only among eligible/dropped rows)
        if "target_trough_v2" in test_data.columns:
            valid = test_data[test_data["target_trough_v2"].notna()]
            if len(valid) > 0:
                sig_on = valid[valid["signal"] == 1]
                total_signals = len(sig_on)
                precision = float(sig_on["target_trough_v2"].mean()) if total_signals > 0 else 0
                total_troughs = int(valid["target_trough_v2"].sum())
                recall = float(sig_on["target_trough_v2"].sum() / total_troughs) if total_troughs > 0 else 0

                all_metrics.append({
                    "fold": fold_id,
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "eligible_rows": len(valid),
                    "total_signals": total_signals,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0,
                    "trough_rate": float(valid["target_trough_v2"].mean()),
                    "dropped_in_test": int(test_data["is_dropped"].sum()),
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
                exit_by_profit = ret_now >= take_profit

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

    # Summary
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
        "fold_metrics": all_metrics,
        "trade_summary": trade_summary,
        "trades": all_trades,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="Rule-based Trough Detection v2")
    ap.add_argument("--data", default="data/features_lgbm.csv")
    ap.add_argument("--out-analysis", default="rule_v2_analysis.json")
    ap.add_argument("--out-trades", default="rule_v2_trades.json")
    ap.add_argument("--out-dashboard", default="rule_v2_dashboard.csv")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    print("=" * 60)
    print("Rule-Based Trough Detection v2")
    print("  타겟: '크게 하락한 종목이 반등하는가?'")
    print("=" * 60)

    df = load_data(csv_path)
    df = add_features(df)
    print(f"Data: {len(df):,} rows, {df['stock_code'].nunique()} stocks\n")

    # ── Test multiple target definitions ──
    target_configs = [
        {"label": "A: drop≥10%, bounce≥5%/10d",
         "drop": 0.10, "bounce_horizon": 10, "bounce_threshold": 0.05},
        {"label": "B: drop≥15%, bounce≥5%/10d",
         "drop": 0.15, "bounce_horizon": 10, "bounce_threshold": 0.05},
        {"label": "C: drop≥10%, bounce≥7%/10d",
         "drop": 0.10, "bounce_horizon": 10, "bounce_threshold": 0.07},
    ]

    print("=" * 60)
    print("Phase 1: Target Distribution Analysis")
    print("=" * 60)

    all_insights = {}
    for cfg in target_configs:
        df_tmp = add_target_v2(df.copy(), cfg["drop"], cfg["bounce_horizon"], cfg["bounce_threshold"])
        eligible = df_tmp["target_trough_v2"].notna().sum()
        positive = df_tmp["target_trough_v2"].dropna().sum()
        pct = positive / eligible * 100 if eligible > 0 else 0
        print(f"\n  {cfg['label']}:")
        print(f"    Eligible (dropped): {eligible:,} / {len(df):,} ({eligible/len(df)*100:.1f}%)")
        print(f"    Bounced: {int(positive):,} ({pct:.1f}%)")
        ins = analyze_trough_distribution(df_tmp, cfg["label"])
        all_insights[cfg["label"]] = ins

    # ── Use config A as main, run strategies ──
    main_cfg = target_configs[0]
    df = add_target_v2(df, main_cfg["drop"], main_cfg["bounce_horizon"], main_cfg["bounce_threshold"])

    print(f"\n\n{'='*60}")
    print(f"Phase 2: Rule Strategies (target: {main_cfg['label']})")
    print(f"{'='*60}")

    strategies = [
        TroughRuleV2(
            drop_threshold=0.10,
            rsi_max=35, bb_max=0.15, disp_max=96,
            hist_diff_positive=True, vol_percentile=0.60,
            min_conditions=2,
            name="conservative",
        ),
        TroughRuleV2(
            drop_threshold=0.10,
            rsi_max=40, bb_max=0.25, disp_max=98,
            hist_diff_positive=True, vol_percentile=0.50,
            min_conditions=2,
            name="balanced",
        ),
        TroughRuleV2(
            drop_threshold=0.10,
            rsi_max=45, bb_max=0.35, disp_max=99,
            hist_diff_positive=True, vol_percentile=0.40,
            min_conditions=2,
            name="aggressive",
        ),
        # MACD 반전 중심 전략
        TroughRuleV2(
            drop_threshold=0.10,
            rsi_max=50, bb_max=0.50, disp_max=100,
            hist_diff_positive=True, vol_percentile=0.30,
            min_conditions=1,  # MACD 반전만으로도 진입
            name="macd_focus",
        ),
    ]

    all_results = []
    for strat in strategies:
        print(f"\n{'─'*50}")
        print(strat.describe())
        print(f"{'─'*50}")

        result = walk_forward_backtest(
            df, strat,
            take_profit=main_cfg["bounce_threshold"],
        )
        all_results.append(result)

        if result["fold_metrics"]:
            mdf = pd.DataFrame(result["fold_metrics"])
            print(f"\n  Folds: {len(mdf)}")
            print(f"  Eligible rows/fold: {mdf['eligible_rows'].mean():.0f}")
            print(f"  Signals/fold:  {mdf['total_signals'].mean():.1f}")
            print(f"  Precision:     {mdf['precision'].mean():.4f} ± {mdf['precision'].std():.4f}")
            print(f"  Recall:        {mdf['recall'].mean():.4f} ± {mdf['recall'].std():.4f}")
            print(f"  F1:            {mdf['f1'].mean():.4f} ± {mdf['f1'].std():.4f}")

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

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<15s} {'Trades':>7s} {'WinRate':>8s} {'AvgRet':>8s} {'Prec':>8s} "
          f"{'Recall':>8s} {'F1':>8s} {'CumRet':>10s} {'MDD':>8s}")
    print("-" * 90)

    for r in all_results:
        ts = r["trade_summary"]
        trades = ts.get("total_trades", 0)
        wr = ts.get("win_rate", 0)
        ar = ts.get("avg_return", 0)
        cr = ts.get("cumulative_return", 0)
        mdd = ts.get("max_drawdown", 0)
        if r["fold_metrics"]:
            mdf = pd.DataFrame(r["fold_metrics"])
            prec = mdf["precision"].mean()
            rec = mdf["recall"].mean()
            f1 = mdf["f1"].mean()
        else:
            prec = rec = f1 = 0

        print(f"{r['strategy']:<15s} {trades:>7d} {wr:>8.4f} {ar:>8.4f} {prec:>8.4f} "
              f"{rec:>8.4f} {f1:>8.4f} {cr:>10.4f} {mdd:>8.4f}")

    # ── Save ──
    output = {
        "config": {"main_target": main_cfg, "all_targets": target_configs},
        "distribution_insights": {k: v for k, v in all_insights.items()},
        "strategies": [
            {"name": r["strategy"], "fold_metrics": r["fold_metrics"],
             "trade_summary": r["trade_summary"]}
            for r in all_results
        ],
    }
    Path(args.out_analysis).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    all_trades = []
    for r in all_results:
        for t in r["trades"]:
            t["strategy"] = r["strategy"]
            all_trades.append(t)
    Path(args.out_trades).write_text(
        json.dumps(all_trades, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ── Dashboard CSV ──
    # 모든 전략의 시그널을 df에 병합하여 대시보드용 CSV 생성
    print(f"\n{'─'*60}")
    print("Generating dashboard CSV...")

    dash = df[["stock_code", "date", "종가"]].copy()
    dash = dash.rename(columns={"종가": "close"})

    # 종목명 (있으면 추가)
    if "종목명" in df.columns:
        dash["stock_name"] = df["종목명"]

    # 핵심 피처들
    feat_cols = ["ret_1", "ret_5", "ret_20", "RSI_14", "BB_pctB",
                 "disparity_20", "volatility_20", "macd_hist", "hist_diff"]
    for col in feat_cols:
        if col in df.columns:
            dash[col] = df[col]

    # 시장 지표
    for col in ["index_ret_5", "index_ret_20"]:
        if col in df.columns:
            dash[col] = df[col]

    # 수급 (있으면)
    for col in ["외국인_비중", "기관_비중", "수급_불균형"]:
        if col in df.columns:
            dash[col] = df[col]

    # 타겟 & 상태
    dash["is_dropped"] = df["is_dropped"]                    # 하락 필터 통과 여부
    dash["target_trough"] = df["target_trough_v2"]           # 실제 반등 여부 (NaN=비대상)
    dash["fwd_max_rally"] = df["fwd_max_rally"]              # 향후 실제 최대 상승폭

    # 각 전략별 시그널 컬럼
    for strat in strategies:
        sig = strat.generate_signals(df)
        dash[f"signal_{strat.name}"] = sig.values

    # 조건 충족 수 (balanced 기준 세부 조건)
    cond = pd.DataFrame(index=df.index)
    if "RSI_14" in df.columns:
        cond["rsi_under40"] = (df["RSI_14"] < 40).astype(int)
    if "BB_pctB" in df.columns:
        cond["bb_under025"] = (df["BB_pctB"] < 0.25).astype(int)
    if "disparity_20" in df.columns:
        cond["disp_under98"] = (df["disparity_20"] < 98).astype(int)
    if "hist_diff" in df.columns:
        cond["macd_turning_up"] = (df["hist_diff"] > 0).astype(int)
    if "volatility_20" in df.columns:
        vol_rank = df.groupby("stock_code")["volatility_20"].rank(pct=True)
        cond["vol_high"] = (vol_rank > 0.50).astype(int)
    dash["conditions_met"] = cond.sum(axis=1)

    # 거래 매핑: 해당 날짜에 어떤 전략이 진입/청산했는지
    for r in all_results:
        sname = r["strategy"]
        entry_col = f"trade_entry_{sname}"
        exit_col = f"trade_exit_{sname}"
        ret_col = f"trade_return_{sname}"
        dash[entry_col] = 0
        dash[exit_col] = 0
        dash[ret_col] = np.nan

        for t in r["trades"]:
            entry_mask = (dash["stock_code"] == t["stock_code"]) & \
                         (dash["date"] == pd.Timestamp(t["entry_date"]))
            exit_mask = (dash["stock_code"] == t["stock_code"]) & \
                        (dash["date"] == pd.Timestamp(t["exit_date"]))
            dash.loc[entry_mask, entry_col] = 1
            dash.loc[exit_mask, exit_col] = 1
            dash.loc[exit_mask, ret_col] = t["return"]

    # 날짜 정렬 + 저장
    dash = dash.sort_values(["stock_code", "date"]).reset_index(drop=True)
    dash["date"] = dash["date"].dt.strftime("%Y-%m-%d")
    dash.to_csv(args.out_dashboard, index=False, encoding="utf-8-sig")

    n_signals = {s.name: int(dash[f"signal_{s.name}"].sum()) for s in strategies}
    print(f"  Rows: {len(dash):,}")
    print(f"  Columns: {len(dash.columns)}")
    print(f"  Signals per strategy: {n_signals}")
    print(f"  Saved: {args.out_dashboard}")

    print(f"\nSaved: {args.out_analysis}, {args.out_trades}, {args.out_dashboard}")
    print(f"[TIME] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()