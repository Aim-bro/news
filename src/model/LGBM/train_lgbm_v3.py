#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM v3 - Single-model, price + supply-demand features only.

Changes from v2:
- Removed all news features and two-stage architecture
- Single LightGBM model per horizon (no Stage1/Stage2 split)
- Feature auto-selection retained (importance-based after warmup fold)
- Cleaner codebase, faster training

Input:  ./data/features_with_news.csv
Output: summary.json, trades.json, feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("lightgbm is required. Install with `pip install lightgbm`.") from e

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


# ─── Constants ───────────────────────────────────────────────────────────────

ID_COLS = ["종목코드", "종목명", "날짜", "date", "stock_code"]
PRICE_LEVEL_COLS = [
    "종가", "최고가", "최저가",
    "index_kospi_close", "index_kospi_high", "index_kospi_low",
    "index_close", "index_high", "index_low",
]
# Explicitly excluded - no longer used
NEWS_COLS = [
    "news_count",
    "sent_mean", "sent_pos_ratio", "sent_neg_ratio",
    "sent_max", "sent_min", "sent_3ma", "sent_momentum",
    "has_news",
]

TARGET_THRESHOLDS = [0.02, 0.03, 0.05]

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=10,
    min_split_gain=1e-4,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.1,
    force_col_wise=True,
    verbosity=-1,
    random_state=42,
    n_jobs=-1,
)

FEATURE_SELECTION = dict(
    warmup_folds=1,
    min_features=15,
    importance_pct=0.85,
)


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Fold:
    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


@dataclass
class FeatureTracker:
    importance_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    importance_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fold_records: List[Dict] = field(default_factory=list)

    def update(self, model: lgb.LGBMRegressor, feature_names: List[str],
               fold_id: int, horizon: int):
        imp = model.feature_importances_
        for fname, val in zip(feature_names, imp):
            self.importance_sum[fname] += float(val)
            self.importance_count[fname] += 1
            self.fold_records.append({
                "fold": fold_id, "horizon": horizon,
                "feature": fname, "importance": float(val),
            })

    def get_top_features(self, all_features: List[str],
                         min_features: int, cum_pct: float) -> List[str]:
        if not self.importance_sum:
            return all_features

        avg_imp = {f: self.importance_sum[f] / max(1, self.importance_count[f])
                   for f in all_features if f in self.importance_sum}
        if not avg_imp:
            return all_features

        sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_feats)
        if total == 0:
            return all_features

        selected = []
        running = 0.0
        for fname, val in sorted_feats:
            selected.append(fname)
            running += val
            if running / total >= cum_pct and len(selected) >= min_features:
                break

        if len(selected) < min_features:
            for fname, _ in sorted_feats:
                if fname not in selected:
                    selected.append(fname)
                    if len(selected) >= min_features:
                        break
        return selected

    def to_dataframe(self) -> pd.DataFrame:
        if not self.fold_records:
            return pd.DataFrame()
        return pd.DataFrame(self.fold_records)

    def summary(self, all_features: List[str]) -> pd.DataFrame:
        rows = []
        for f in all_features:
            cnt = self.importance_count.get(f, 0)
            if cnt > 0:
                rows.append({
                    "feature": f,
                    "avg_importance": self.importance_sum[f] / cnt,
                    "total_importance": self.importance_sum[f],
                    "fold_count": cnt,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("avg_importance", ascending=False).reset_index(drop=True)


# ─── Utilities ───────────────────────────────────────────────────────────────

def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if spearmanr is None:
        s1 = pd.Series(y_true).rank()
        s2 = pd.Series(y_pred).rank()
        return float(s1.corr(s2))
    val = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(val) if val is not None else np.nan


def _to_jsonable(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if pd.isna(v):
        return None
    return v


# ─── Data loading ────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "날짜" in df.columns:
        df["date"] = pd.to_datetime(df["날짜"], errors="coerce")
    else:
        raise ValueError("No date column found ('date' or '날짜').")

    if "stock_code" in df.columns:
        code_src = df["stock_code"]
    elif "종목코드" in df.columns:
        code_src = df["종목코드"]
    else:
        raise ValueError("No stock code column found ('stock_code' or '종목코드').")

    df["stock_code"] = (
        code_src.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(6)
    )
    if "종목코드" not in df.columns:
        df["종목코드"] = df["stock_code"]

    skip = set(ID_COLS) | {"date"}
    for c in df.columns:
        if c not in skip:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="ignore")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return df


def add_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)
    g = out.groupby("stock_code", sort=False)
    for n in horizons:
        out[f"fwd_{n}"] = g["종가"].shift(-n) / out["종가"] - 1.0
    return out


def build_features(df: pd.DataFrame) -> List[str]:
    """Build feature list: all numeric cols except IDs, targets, price levels, and news."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [c for c in df.columns if c.startswith("fwd_")]
    exclude = set(ID_COLS) | {"date"} | set(target_cols) | set(PRICE_LEVEL_COLS) | set(NEWS_COLS)

    features = [c for c in numeric_cols if c not in exclude]
    if not features:
        raise ValueError("No features found after exclusion.")
    return features


# ─── Walk-forward folds ─────────────────────────────────────────────────────

def make_folds(
    df: pd.DataFrame,
    initial_train_end: str,
    test_window_days: int,
    embargo_days: int = 0,
) -> List[Fold]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) == 0:
        return []

    init_end = pd.Timestamp(initial_train_end)
    train_end_idx = np.searchsorted(dates, init_end.to_datetime64(), side="right") - 1
    if train_end_idx < 20:
        train_end_idx = max(20, int(len(dates) * 0.6))
    train_end_idx = min(train_end_idx, len(dates) - 2)

    folds: List[Fold] = []
    fold_id = 1
    while True:
        test_start_idx = train_end_idx + 1 + embargo_days
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_start_idx + test_window_days - 1, len(dates) - 1)

        folds.append(Fold(
            fold_id=fold_id,
            train_end_date=pd.Timestamp(dates[train_end_idx]),
            test_start_date=pd.Timestamp(dates[test_start_idx]),
            test_end_date=pd.Timestamp(dates[test_end_idx]),
        ))
        fold_id += 1
        train_end_idx = test_end_idx
        if train_end_idx >= len(dates) - 2:
            break

    return folds


# ─── Model ───────────────────────────────────────────────────────────────────

def _train_valid_split(train_df: pd.DataFrame, min_valid_dates: int = 20):
    udates = np.array(sorted(train_df["date"].dropna().unique()))
    if len(udates) < max(10, min_valid_dates + 10):
        valid_size = max(5, len(udates) // 5)
    else:
        valid_size = max(min_valid_dates, len(udates) // 5)
    valid_size = min(valid_size, max(1, len(udates) - 1))

    valid_start = pd.Timestamp(udates[-valid_size])
    tr_mask = (train_df["date"] < valid_start).values
    va_mask = (train_df["date"] >= valid_start).values
    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        cut = max(1, int(len(train_df) * 0.8))
        tr_mask = np.zeros(len(train_df), dtype=bool)
        tr_mask[:cut] = True
        va_mask = ~tr_mask
    return tr_mask, va_mask


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {k: np.nan for k in ["rmse", "mae", "r2", "dir_acc", "spearman_ic"]}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    ic = _safe_spearman(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc, spearman_ic=ic)


def train_fold(
    fold_train: pd.DataFrame,
    fold_test: pd.DataFrame,
    target_col: str,
    features: List[str],
    tracker: FeatureTracker,
    fold_id: int,
    horizon: int,
) -> Tuple[np.ndarray, lgb.LGBMRegressor]:
    tr_mask, va_mask = _train_valid_split(fold_train)
    train_part = fold_train.iloc[np.where(tr_mask)[0]]
    valid_part = fold_train.iloc[np.where(va_mask)[0]]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        train_part[features], train_part[target_col],
        eval_set=[(valid_part[features], valid_part[target_col])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    tracker.update(model, features, fold_id, horizon)

    y_pred = model.predict(fold_test[features], num_iteration=model.best_iteration_)
    return y_pred, model


# ─── Backtest ────────────────────────────────────────────────────────────────

def run_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> Tuple[List[dict], Dict[str, float]]:
    work = pred_df[pred_df["horizon"] == horizon].copy()
    work = work.sort_values(["stock_code", "date"]).reset_index(drop=True)

    trades: List[dict] = []
    for stock_code, g in work.groupby("stock_code", sort=False):
        pos = None
        for _, row in g.iterrows():
            px = float(row["종가"])
            pred = float(row["pred"])
            dt = pd.Timestamp(row["date"])

            if pos is None:
                if pred > threshold:
                    pos = {
                        "stock_code": stock_code,
                        "stock_name": row.get("종목명", None),
                        "entry_date": dt,
                        "entry_price": px,
                        "entry_pred": pred,
                        "bars": 0,
                    }
                continue

            pos["bars"] += 1
            ret_now = px / pos["entry_price"] - 1.0
            exit_by_signal = pred < -(threshold * 0.5)
            exit_by_time = pos["bars"] > (2 * horizon)
            exit_by_stop = ret_now <= -0.15

            if exit_by_signal or exit_by_time or exit_by_stop:
                trades.append({
                    "horizon": horizon,
                    "threshold": threshold,
                    "stock_code": pos["stock_code"],
                    "stock_name": pos["stock_name"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": px,
                    "return": ret_now,
                    "holding_days": pos["bars"],
                    "exit_reason": "signal" if exit_by_signal else ("stop" if exit_by_stop else "time"),
                    "entry_pred": pos["entry_pred"],
                    "exit_pred": pred,
                })
                pos = None

    if not trades:
        return [], {
            "total_trades": 0, "win_rate": np.nan, "cumulative_return": np.nan,
            "wl_ratio": np.nan, "avg_holding_days": np.nan, "max_drawdown": np.nan,
        }

    tdf = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    equity = (1.0 + tdf["return"].astype(float)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    wins = tdf[tdf["return"] > 0]["return"]
    losses = tdf[tdf["return"] <= 0]["return"]
    wl_ratio = np.nan
    if len(wins) and len(losses):
        wl_ratio = float(wins.mean() / abs(losses.mean())) if losses.mean() != 0 else np.nan

    summary = {
        "total_trades": int(len(tdf)),
        "win_rate": float((tdf["return"] > 0).mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "wl_ratio": wl_ratio,
        "avg_holding_days": float(tdf["holding_days"].mean()),
        "max_drawdown": float(dd.min()),
    }
    return trades, summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser(description="LightGBM v3 - single model, price+supply features")
    ap.add_argument("--data", default="data/features_with_news.csv", help="Input CSV path")
    ap.add_argument("--initial-train-end", default="2024-06-30")
    ap.add_argument("--test-window-days", type=int, default=60)
    ap.add_argument("--embargo-days", type=int, default=0,
                    help="Gap between train/test (set to max horizon for strict leakage prevention)")
    ap.add_argument("--auto-select", action="store_true", default=True)
    ap.add_argument("--no-auto-select", dest="auto_select", action="store_false")
    ap.add_argument("--out-summary", default="summary.json")
    ap.add_argument("--out-trades", default="trades.json")
    ap.add_argument("--out-importance", default="feature_importance.csv")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    horizons = [5, 10, 20]

    print("=" * 60)
    print("LightGBM v3 - Single model, price + supply-demand features")
    print("=" * 60)
    print(f"Loading: {csv_path}")

    t0_load = time.perf_counter()
    df = load_and_prepare(csv_path)
    df = add_targets(df, horizons)
    all_features = build_features(df)
    t1_load = time.perf_counter()

    print(f"Rows: {len(df):,}, Stocks: {df['stock_code'].nunique():,}")
    print(f"Features: {len(all_features)}")
    print(f"Excluded news cols: {NEWS_COLS}")
    print(f"Embargo: {args.embargo_days} days | Auto-select: {args.auto_select}")
    print(f"[TIME] data prep: {t1_load - t0_load:.2f}s\n")

    # Print feature list grouped by category
    supply_feats = [f for f in all_features if any(k in f for k in ["외국인", "기관", "개인", "수급"])]
    price_feats = [f for f in all_features if f not in supply_feats]
    print(f"Price/technical features ({len(price_feats)}): {price_feats[:10]}{'...' if len(price_feats) > 10 else ''}")
    print(f"Supply/demand features ({len(supply_feats)}): {supply_feats[:10]}{'...' if len(supply_feats) > 10 else ''}")
    print()

    folds = make_folds(df, args.initial_train_end, args.test_window_days, args.embargo_days)
    if not folds:
        raise SystemExit("No walk-forward folds generated.")

    print(f"Folds: {len(folds)}")
    for f in folds:
        print(f"  F{f.fold_id}: train <= {f.train_end_date.date()} | "
              f"test {f.test_start_date.date()} ~ {f.test_end_date.date()}")

    tracker = FeatureTracker()
    all_fold_metrics: List[dict] = []
    all_preds: List[pd.DataFrame] = []

    t0_train = time.perf_counter()
    for horizon in horizons:
        t0_h = time.perf_counter()
        target_col = f"fwd_{horizon}"
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon} days")
        print(f"{'='*60}")

        base_cols = ["stock_code", "종목명", "date", "종가", target_col] + all_features
        base_cols = list(dict.fromkeys([c for c in base_cols if c in df.columns]))
        ds = df[base_cols].copy()

        mask_clean = ds[all_features + [target_col]].notna().all(axis=1)
        print(f"Rows after NA drop: {int(mask_clean.sum()):,} / {len(ds):,}")
        ds_clean = ds.loc[mask_clean].copy().reset_index(drop=True)

        if len(ds_clean) == 0:
            print("Skipped: no rows")
            continue

        active_features = all_features.copy()

        for fold in folds:
            tr = ds_clean.loc[ds_clean["date"] <= fold.train_end_date]
            te = ds_clean.loc[
                (ds_clean["date"] >= fold.test_start_date) &
                (ds_clean["date"] <= fold.test_end_date)
            ]

            if len(tr) < 500 or len(te) == 0:
                print(f"  [F{fold.fold_id}] skipped (train={len(tr)}, test={len(te)})")
                continue

            # Auto feature selection after warmup
            if (args.auto_select
                    and fold.fold_id > FEATURE_SELECTION["warmup_folds"]
                    and tracker.importance_sum):
                selected = tracker.get_top_features(
                    all_features,
                    FEATURE_SELECTION["min_features"],
                    FEATURE_SELECTION["importance_pct"],
                )
                if len(selected) < len(all_features):
                    print(f"  [F{fold.fold_id}] Auto-select: {len(selected)}/{len(all_features)} features")
                active_features = selected
            else:
                active_features = all_features.copy()

            print(f"  [H{horizon}][F{fold.fold_id}] train={len(tr):,} test={len(te):,} feats={len(active_features)}")

            y_pred, model = train_fold(
                tr, te, target_col,
                active_features, tracker, fold.fold_id, horizon,
            )
            y_true = te[target_col].values

            m = calc_metrics(y_true, y_pred)
            all_fold_metrics.append({
                "horizon": horizon,
                "fold": fold.fold_id,
                "train_end": fold.train_end_date,
                "test_start": fold.test_start_date,
                "test_end": fold.test_end_date,
                "n_features": len(active_features),
                "best_iteration": model.best_iteration_,
                **m,
                "rows_test": int(len(te)),
            })
            print(f"    RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f} "
                  f"Dir={m['dir_acc']:.3f} IC={m['spearman_ic']:.3f} "
                  f"(best_iter={model.best_iteration_})")

            pred_rows = te[["stock_code", "종목명", "date", "종가"]].copy()
            pred_rows["horizon"] = horizon
            pred_rows["target"] = y_true
            pred_rows["pred"] = y_pred
            all_preds.append(pred_rows)

        t1_h = time.perf_counter()
        print(f"[TIME] horizon {horizon}: {t1_h - t0_h:.2f}s")

    t1_train = time.perf_counter()
    print(f"\n[TIME] training total: {t1_train - t0_train:.2f}s")

    if not all_fold_metrics or not all_preds:
        raise SystemExit("No predictions produced.")

    # ── Feature importance report ──
    imp_summary = tracker.summary(all_features)
    if len(imp_summary) > 0:
        print(f"\n{'='*60}")
        print("Feature Importance (Top 30)")
        print(f"{'='*60}")
        with pd.option_context("display.max_rows", 30, "display.float_format", "{:.1f}".format):
            print(imp_summary.head(30).to_string(index=False))

        selected_final = tracker.get_top_features(
            all_features,
            FEATURE_SELECTION["min_features"],
            FEATURE_SELECTION["importance_pct"],
        )
        print(f"\nFinal selected features ({len(selected_final)}/{len(all_features)}):")
        for i, f in enumerate(selected_final, 1):
            avg = tracker.importance_sum.get(f, 0) / max(1, tracker.importance_count.get(f, 1))
            cat = "[FLOW]" if any(k in f for k in ["외국인", "기관", "개인", "수급"]) else "[TECH]"
            print(f"  {i:3d}. {cat} {f:<40s} avg_imp={avg:.1f}")

        imp_detail = tracker.to_dataframe()
        if len(imp_detail) > 0:
            imp_detail.to_csv(args.out_importance, index=False, encoding="utf-8-sig")
            print(f"\nSaved: {args.out_importance}")

    # ── Metrics summary ──
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(f"\n{'='*60}")
    print("Per-fold Metrics Summary")
    print(f"{'='*60}")
    for horizon in horizons:
        sub = metrics_df[metrics_df["horizon"] == horizon]
        if len(sub) == 0:
            continue
        print(f"\n  Horizon {horizon}:")
        print(f"    RMSE:  {sub['rmse'].mean():.4f} +/- {sub['rmse'].std():.4f}")
        print(f"    MAE:   {sub['mae'].mean():.4f} +/- {sub['mae'].std():.4f}")
        print(f"    R2:    {sub['r2'].mean():.4f} +/- {sub['r2'].std():.4f}")
        print(f"    Dir:   {sub['dir_acc'].mean():.4f} +/- {sub['dir_acc'].std():.4f}")
        print(f"    IC:    {sub['spearman_ic'].mean():.4f} +/- {sub['spearman_ic'].std():.4f}")

    # ── Backtest ──
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = pred_df.sort_values(["horizon", "stock_code", "date"]).drop_duplicates(
        subset=["horizon", "stock_code", "date"], keep="last"
    )

    t0_bt = time.perf_counter()
    trade_records: List[dict] = []
    summary_rows: List[dict] = []
    for horizon in horizons:
        for th in TARGET_THRESHOLDS:
            trades, bt = run_backtest(pred_df, horizon, th)
            trade_records.extend(trades)

            fold_sub = metrics_df[metrics_df["horizon"] == horizon]
            summary_rows.append({
                "horizon": horizon,
                "threshold": th,
                "folds": int(fold_sub["fold"].nunique()),
                "rmse_mean": float(fold_sub["rmse"].mean()) if len(fold_sub) else np.nan,
                "mae_mean": float(fold_sub["mae"].mean()) if len(fold_sub) else np.nan,
                "r2_mean": float(fold_sub["r2"].mean()) if len(fold_sub) else np.nan,
                "dir_acc_mean": float(fold_sub["dir_acc"].mean()) if len(fold_sub) else np.nan,
                "spearman_ic_mean": float(fold_sub["spearman_ic"].mean()) if len(fold_sub) else np.nan,
                **bt,
            })
    t1_bt = time.perf_counter()
    print(f"\n[TIME] backtest: {t1_bt - t0_bt:.2f}s")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="cumulative_return", ascending=False, na_position="last"
    ).reset_index(drop=True)

    show_cols = [
        "horizon", "threshold",
        "total_trades", "win_rate", "cumulative_return", "wl_ratio",
        "avg_holding_days", "max_drawdown",
        "rmse_mean", "mae_mean", "r2_mean", "dir_acc_mean", "spearman_ic_mean",
    ]
    print(f"\n{'='*60}")
    print("Ranked Configurations (by cumulative_return)")
    print(f"{'='*60}")
    with pd.option_context(
        "display.max_rows", 200, "display.max_columns", None,
        "display.width", 250, "display.float_format", "{:.4f}".format
    ):
        print(summary_df[show_cols])

    # ── Save ──
    out_summary = {
        "config": {
            "data": str(csv_path),
            "initial_train_end": args.initial_train_end,
            "test_window_days": args.test_window_days,
            "embargo_days": args.embargo_days,
            "horizons": horizons,
            "thresholds": TARGET_THRESHOLDS,
            "lgb_params": LGB_PARAMS,
            "feature_selection": FEATURE_SELECTION if args.auto_select else "disabled",
            "total_features": len(all_features),
            "excluded_news_cols": NEWS_COLS,
        },
        "fold_metrics": [{k: _to_jsonable(v) for k, v in r.items()} for r in all_fold_metrics],
        "summary": [{k: _to_jsonable(v) for k, v in r.items()} for r in summary_df.to_dict(orient="records")],
    }
    out_trades = [{k: _to_jsonable(v) for k, v in r.items()} for r in trade_records]

    Path(args.out_summary).write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_trades).write_text(json.dumps(out_trades, ensure_ascii=False, indent=2), encoding="utf-8")

    t1_total = time.perf_counter()
    print(f"\nSaved: {args.out_summary}, {args.out_trades}, {args.out_importance}")
    print(f"[TIME] end-to-end: {t1_total - t0_total:.2f}s")


if __name__ == "__main__":
    main()
