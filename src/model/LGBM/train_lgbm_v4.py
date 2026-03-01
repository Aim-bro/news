#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM v4.1 — Optuna tuned, adapted to features_lgbm.csv (30 cols).

Data: 15,310 rows, 21 stocks, 30 columns (price + supply-demand + returns).
NA: minimal (max 2.7% from 20-day warmup) — filled with 0 to maximize rows.
No news features, no two-stage.

Pipeline:
1. Load & prepare (fillna for warmup NAs)
2. Build forward-return targets
3. [TUNE] Optuna per horizon on first fold's training data
4. [EVAL] Walk-forward with tuned params + auto feature selection
5. Report prediction metrics + feature importance

Input:  ./data/features_lgbm.csv
Output: summary.json, trades.json, feature_importance.csv, best_params.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("lightgbm required: pip install lightgbm") from e

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna required: pip install optuna") from e

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")


# ─── Constants ───────────────────────────────────────────────────────────────

ID_COLS = ["종목코드", "종목명", "날짜", "date", "stock_code"]
PRICE_LEVEL_COLS = [
    "종가", "최고가", "최저가",
    "index_kospi_close", "index_kospi_high", "index_kospi_low",
]
TARGET_THRESHOLDS = [0.02, 0.03, 0.05]

DEFAULT_LGB_PARAMS = dict(
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
    min_features=10,       # fewer total features, lower minimum
    importance_pct=0.90,
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
        selected, running = [], 0.0
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
        return pd.DataFrame(self.fold_records) if self.fold_records else pd.DataFrame()

    def summary(self, all_features: List[str]) -> pd.DataFrame:
        rows = []
        for f in all_features:
            cnt = self.importance_count.get(f, 0)
            if cnt > 0:
                rows.append({
                    "feature": f, "avg_importance": self.importance_sum[f] / cnt,
                    "total_importance": self.importance_sum[f], "fold_count": cnt,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("avg_importance", ascending=False).reset_index(drop=True)


# ─── Utilities ───────────────────────────────────────────────────────────────

def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return np.nan
    if spearmanr is None:
        return float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))
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


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {k: np.nan for k in ["rmse", "mae", "r2", "dir_acc", "spearman_ic"]}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    ic = _safe_spearman(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc, spearman_ic=ic)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "날짜" in df.columns:
        df["date"] = pd.to_datetime(df["날짜"], errors="coerce")
    else:
        raise ValueError("No date column found.")

    # Stock code
    if "stock_code" in df.columns:
        code_src = df["stock_code"]
    elif "종목코드" in df.columns:
        code_src = df["종목코드"]
    else:
        raise ValueError("No stock code column found.")
    df["stock_code"] = code_src.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(6)
    if "종목코드" not in df.columns:
        df["종목코드"] = df["stock_code"]

    # Numeric coercion
    skip = set(ID_COLS) | {"date"}
    for c in df.columns:
        if c not in skip and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # Fill warmup NAs with 0 (ret_1/5/20, index_ret, relative_strength)
    na_before = int(df.isnull().sum().sum())
    fill_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in skip]
    for c in fill_cols:
        df[c] = df[c].fillna(0.0)
    na_after = int(df.isnull().sum().sum())
    print(f"  NA filled: {na_before:,} → {na_after:,} cells")

    return df


def add_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy().sort_values(["stock_code", "date"]).reset_index(drop=True)
    g = out.groupby("stock_code", sort=False)
    for n in horizons:
        out[f"fwd_{n}"] = g["종가"].shift(-n) / out["종가"] - 1.0
    return out


def build_features(df: pd.DataFrame) -> List[str]:
    """All numeric cols except IDs, targets, and raw price levels."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [c for c in df.columns if c.startswith("fwd_")]
    exclude = set(ID_COLS) | {"date"} | set(target_cols) | set(PRICE_LEVEL_COLS)
    features = [c for c in numeric_cols if c not in exclude]
    if not features:
        raise ValueError("No features found.")
    return features


# ─── Walk-forward folds ─────────────────────────────────────────────────────

def make_folds(df: pd.DataFrame, initial_train_end: str,
               test_window_days: int, embargo_days: int = 0) -> List[Fold]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) == 0:
        return []
    init_end = pd.Timestamp(initial_train_end)
    train_end_idx = np.searchsorted(dates, init_end.to_datetime64(), side="right") - 1
    if train_end_idx < 20:
        train_end_idx = max(20, int(len(dates) * 0.6))
    train_end_idx = min(train_end_idx, len(dates) - 2)

    folds = []
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


# ─── Optuna tuning ───────────────────────────────────────────────────────────

def _make_inner_splits(train_df: pd.DataFrame, n_splits: int = 3,
                       embargo_days: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    dates = np.array(sorted(train_df["date"].dropna().unique()))
    n = len(dates)
    if n < n_splits + 2:
        cut_date = pd.Timestamp(dates[int(n * 0.8)])
        tr_idx = np.where(train_df["date"] < cut_date)[0]
        va_idx = np.where(train_df["date"] >= cut_date)[0]
        return [(tr_idx, va_idx)] if len(tr_idx) > 0 and len(va_idx) > 0 else []

    splits = []
    min_train_pct = 0.4
    step = (1.0 - min_train_pct) / (n_splits + 1)
    for i in range(n_splits):
        train_pct = min_train_pct + step * (i + 1)
        cut_idx = int(n * train_pct)
        embargo_end = min(cut_idx + embargo_days, n - 1)
        if embargo_end >= n - 1:
            continue
        valid_end = min(embargo_end + max(int(n * step), 10), n)
        cut_date = pd.Timestamp(dates[cut_idx])
        embargo_date = pd.Timestamp(dates[embargo_end])
        valid_end_date = pd.Timestamp(dates[min(valid_end, n - 1)])
        tr_idx = np.where(train_df["date"] <= cut_date)[0]
        va_idx = np.where(
            (train_df["date"] > embargo_date) & (train_df["date"] <= valid_end_date)
        )[0]
        if len(tr_idx) >= 200 and len(va_idx) >= 50:
            splits.append((tr_idx, va_idx))
    return splits


def _optuna_objective(trial, train_df, features, target_col, inner_splits):
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "force_col_wise": True,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
    }
    params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])

    ic_scores = []
    for tr_idx, va_idx in inner_splits:
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target_col]
        X_va = train_df.iloc[va_idx][features]
        y_va = train_df.iloc[va_idx][target_col]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l2",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        y_pred = model.predict(X_va, num_iteration=model.best_iteration_)
        ic = _safe_spearman(y_va.values, y_pred)
        if not np.isnan(ic):
            ic_scores.append(ic)
    return float(np.mean(ic_scores)) if ic_scores else -1.0


def run_optuna_tuning(train_df, features, target_col,
                      n_trials=80, time_budget_sec=600, embargo_days=0):
    inner_splits = _make_inner_splits(train_df, n_splits=3, embargo_days=embargo_days)
    if not inner_splits:
        print("    [Optuna] Cannot create inner splits, using defaults")
        return DEFAULT_LGB_PARAMS.copy()

    print(f"    [Optuna] Inner CV: {len(inner_splits)} splits, "
          f"sizes: {[(len(t), len(v)) for t, v in inner_splits]}")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42, n_startup_trials=15))
    study.optimize(
        lambda trial: _optuna_objective(trial, train_df, features, target_col, inner_splits),
        n_trials=n_trials, timeout=time_budget_sec, show_progress_bar=False,
    )

    print(f"    [Optuna] {len(study.trials)} trials | Best IC: {study.best_value:.4f}")
    print(f"    [Optuna] Best params: {json.dumps(study.best_params, indent=2)}")

    bp = study.best_params
    best_params = {
        "objective": "regression", "boosting_type": "gbdt",
        "force_col_wise": True, "verbosity": -1, "random_state": 42, "n_jobs": -1,
        **bp,
    }
    best_params["num_leaves"] = min(bp["num_leaves"], 2 ** bp["max_depth"])
    return best_params


# ─── Model training ─────────────────────────────────────────────────────────

def _train_valid_split(train_df, min_valid_dates=20):
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


def train_fold(fold_train, fold_test, target_col, features, lgb_params,
               tracker, fold_id, horizon):
    tr_mask, va_mask = _train_valid_split(fold_train)
    train_part = fold_train.iloc[np.where(tr_mask)[0]]
    valid_part = fold_train.iloc[np.where(va_mask)[0]]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train_part[features], train_part[target_col],
        eval_set=[(valid_part[features], valid_part[target_col])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    tracker.update(model, features, fold_id, horizon)
    y_pred = model.predict(fold_test[features], num_iteration=model.best_iteration_)
    return y_pred, model


# ─── Backtest ────────────────────────────────────────────────────────────────

def run_backtest(pred_df, horizon, threshold):
    work = pred_df[pred_df["horizon"] == horizon].copy()
    work = work.sort_values(["stock_code", "date"]).reset_index(drop=True)
    trades = []
    for stock_code, g in work.groupby("stock_code", sort=False):
        pos = None
        for _, row in g.iterrows():
            px, pred, dt = float(row["종가"]), float(row["pred"]), pd.Timestamp(row["date"])
            if pos is None:
                if pred > threshold:
                    pos = {"stock_code": stock_code, "stock_name": row.get("종목명"),
                           "entry_date": dt, "entry_price": px, "entry_pred": pred, "bars": 0}
                continue
            pos["bars"] += 1
            ret_now = px / pos["entry_price"] - 1.0
            if pred < -(threshold * 0.5) or pos["bars"] > (2 * horizon) or ret_now <= -0.15:
                trades.append({
                    "horizon": horizon, "threshold": threshold,
                    "stock_code": pos["stock_code"], "stock_name": pos["stock_name"],
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "entry_price": pos["entry_price"], "exit_price": px,
                    "return": ret_now, "holding_days": pos["bars"],
                    "exit_reason": "signal" if pred < -(threshold*0.5) else ("stop" if ret_now <= -0.15 else "time"),
                    "entry_pred": pos["entry_pred"], "exit_pred": pred,
                })
                pos = None
    if not trades:
        return [], {"total_trades": 0, "win_rate": np.nan, "cumulative_return": np.nan,
                    "wl_ratio": np.nan, "avg_holding_days": np.nan, "max_drawdown": np.nan}
    tdf = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    equity = (1.0 + tdf["return"].astype(float)).cumprod()
    dd = equity / equity.cummax() - 1.0
    wins, losses = tdf[tdf["return"] > 0]["return"], tdf[tdf["return"] <= 0]["return"]
    wl = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan
    return trades, {
        "total_trades": int(len(tdf)), "win_rate": float((tdf["return"] > 0).mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0), "wl_ratio": wl,
        "avg_holding_days": float(tdf["holding_days"].mean()), "max_drawdown": float(dd.min()),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser(description="LightGBM v4.1 — Optuna tuned (features_lgbm.csv)")
    ap.add_argument("--data", default="data/features_lgbm.csv")
    ap.add_argument("--initial-train-end", default="2024-06-30")
    ap.add_argument("--test-window-days", type=int, default=60)
    ap.add_argument("--embargo-days", type=int, default=0)
    ap.add_argument("--auto-select", action="store_true", default=True)
    ap.add_argument("--no-auto-select", dest="auto_select", action="store_false")
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--tune-budget", type=int, default=600)
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--load-params", type=str, default=None)
    ap.add_argument("--out-summary", default="summary.json")
    ap.add_argument("--out-trades", default="trades.json")
    ap.add_argument("--out-importance", default="feature_importance.csv")
    ap.add_argument("--out-params", default="best_params.json")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    horizons = [5, 10, 20]

    print("=" * 60)
    print("LightGBM v4.1 — Optuna tuned (features_lgbm.csv)")
    print("=" * 60)
    print(f"Data: {csv_path}")

    t0_load = time.perf_counter()
    df = load_and_prepare(csv_path)
    df = add_targets(df, horizons)
    all_features = build_features(df)
    t1_load = time.perf_counter()

    print(f"Rows: {len(df):,}, Stocks: {df['stock_code'].nunique():,}")
    print(f"Features ({len(all_features)}): {all_features}")
    print(f"Embargo: {args.embargo_days} days | Auto-select: {args.auto_select}")
    print(f"Optuna: {args.n_trials} trials/horizon, {args.tune_budget}s budget/horizon")
    print(f"[TIME] data prep: {t1_load - t0_load:.2f}s\n")

    folds = make_folds(df, args.initial_train_end, args.test_window_days, args.embargo_days)
    if not folds:
        raise SystemExit("No folds generated.")
    print(f"Folds: {len(folds)}")
    for f in folds:
        print(f"  F{f.fold_id}: train ≤ {f.train_end_date.date()} | "
              f"test {f.test_start_date.date()} ~ {f.test_end_date.date()}")

    # ── Hyperparameter tuning ──
    best_params_per_horizon: Dict[int, Dict] = {}

    if args.load_params:
        loaded = json.loads(Path(args.load_params).read_text(encoding="utf-8"))
        for h in horizons:
            best_params_per_horizon[h] = loaded.get(str(h), DEFAULT_LGB_PARAMS.copy())
            print(f"\n[H{h}] Loaded params from {args.load_params}")
    elif args.skip_tune:
        for h in horizons:
            best_params_per_horizon[h] = DEFAULT_LGB_PARAMS.copy()
        print("\nSkipping Optuna, using defaults")
    else:
        print(f"\n{'='*60}")
        print("Phase 1: Optuna Hyperparameter Tuning")
        print(f"{'='*60}")

        for horizon in horizons:
            t0_tune = time.perf_counter()
            target_col = f"fwd_{horizon}"
            print(f"\n--- Horizon {horizon} ---")

            tune_end = folds[0].train_end_date
            ds = df[["stock_code", "date", "종가", target_col] + all_features].copy()
            mask = ds[target_col].notna()
            tune_data = ds.loc[mask & (ds["date"] <= tune_end)].reset_index(drop=True)
            print(f"  Tune data: {len(tune_data):,} rows (≤ {tune_end.date()})")

            if len(tune_data) < 500:
                print("  Too few rows, using defaults")
                best_params_per_horizon[horizon] = DEFAULT_LGB_PARAMS.copy()
            else:
                best_params_per_horizon[horizon] = run_optuna_tuning(
                    tune_data, all_features, target_col,
                    n_trials=args.n_trials, time_budget_sec=args.tune_budget,
                    embargo_days=args.embargo_days,
                )
            print(f"  [TIME] tune H{horizon}: {time.perf_counter() - t0_tune:.1f}s")

        Path(args.out_params).write_text(
            json.dumps({str(h): p for h, p in best_params_per_horizon.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.out_params}")

    # ── Walk-forward evaluation ──
    print(f"\n{'='*60}")
    print("Phase 2: Walk-Forward Evaluation")
    print(f"{'='*60}")

    tracker = FeatureTracker()
    all_fold_metrics, all_preds = [], []

    t0_train = time.perf_counter()
    for horizon in horizons:
        t0_h = time.perf_counter()
        target_col = f"fwd_{horizon}"
        lgb_params = best_params_per_horizon[horizon]

        print(f"\n{'─'*50}")
        print(f"Horizon: {horizon} days")
        print(f"  lr={lgb_params.get('learning_rate', '?'):.4f}, "
              f"n_est={lgb_params.get('n_estimators', '?')}, "
              f"depth={lgb_params.get('max_depth', '?')}, "
              f"leaves={lgb_params.get('num_leaves', '?')}, "
              f"min_child={lgb_params.get('min_child_samples', '?')}")
        print(f"{'─'*50}")

        ds = df[["stock_code", "종목코드", "date", "종가", target_col] + all_features].copy()
        mask_target = ds[target_col].notna()
        ds_clean = ds.loc[mask_target].reset_index(drop=True)
        print(f"  Rows (target valid): {len(ds_clean):,}")

        if len(ds_clean) == 0:
            continue

        active_features = all_features.copy()

        for fold in folds:
            tr = ds_clean.loc[ds_clean["date"] <= fold.train_end_date]
            te = ds_clean.loc[
                (ds_clean["date"] >= fold.test_start_date) &
                (ds_clean["date"] <= fold.test_end_date)
            ]
            if len(tr) < 500 or len(te) == 0:
                continue

            if (args.auto_select
                    and fold.fold_id > FEATURE_SELECTION["warmup_folds"]
                    and tracker.importance_sum):
                selected = tracker.get_top_features(
                    all_features, FEATURE_SELECTION["min_features"],
                    FEATURE_SELECTION["importance_pct"],
                )
                if len(selected) < len(all_features):
                    print(f"  [F{fold.fold_id}] Auto-select: {len(selected)}/{len(all_features)}")
                active_features = selected
            else:
                active_features = all_features.copy()

            print(f"  [H{horizon}][F{fold.fold_id}] train={len(tr):,} test={len(te):,} "
                  f"feats={len(active_features)}")

            y_pred, model = train_fold(
                tr, te, target_col, active_features,
                lgb_params, tracker, fold.fold_id, horizon,
            )
            y_true = te[target_col].values
            m = calc_metrics(y_true, y_pred)

            all_fold_metrics.append({
                "horizon": horizon, "fold": fold.fold_id,
                "train_end": fold.train_end_date, "test_start": fold.test_start_date,
                "test_end": fold.test_end_date, "n_features": len(active_features),
                "best_iteration": model.best_iteration_, **m, "rows_test": int(len(te)),
            })
            print(f"    RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R²={m['r2']:.4f} "
                  f"Dir={m['dir_acc']:.3f} IC={m['spearman_ic']:.3f} "
                  f"(iter={model.best_iteration_})")

            pred_rows = te[["stock_code", "종목코드", "date", "종가"]].copy()
            pred_rows["horizon"] = horizon
            pred_rows["target"] = y_true
            pred_rows["pred"] = y_pred
            all_preds.append(pred_rows)

        print(f"  [TIME] H{horizon}: {time.perf_counter() - t0_h:.2f}s")

    print(f"\n[TIME] evaluation: {time.perf_counter() - t0_train:.2f}s")

    if not all_fold_metrics:
        raise SystemExit("No predictions produced.")

    # ── Feature importance ──
    imp_summary = tracker.summary(all_features)
    if len(imp_summary) > 0:
        print(f"\n{'='*60}")
        print("Feature Importance (all features)")
        print(f"{'='*60}")
        with pd.option_context("display.max_rows", 50, "display.float_format", "{:.1f}".format):
            print(imp_summary.to_string(index=False))

        selected_final = tracker.get_top_features(
            all_features, FEATURE_SELECTION["min_features"], FEATURE_SELECTION["importance_pct"],
        )
        print(f"\nFinal features ({len(selected_final)}/{len(all_features)}):")
        for i, f in enumerate(selected_final, 1):
            avg = tracker.importance_sum.get(f, 0) / max(1, tracker.importance_count.get(f, 1))
            cat = "📊수급" if any(k in f for k in ["외국인", "기관", "개인", "수급"]) else "📈기술"
            print(f"  {i:3d}. {cat} {f:<40s} imp={avg:.1f}")

        tracker.to_dataframe().to_csv(args.out_importance, index=False, encoding="utf-8-sig")

    # ── Prediction metrics ──
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(f"\n{'='*60}")
    print("Prediction Performance (mean ± std)")
    print(f"{'='*60}")
    for horizon in horizons:
        sub = metrics_df[metrics_df["horizon"] == horizon]
        if len(sub) == 0:
            continue
        print(f"\n  Horizon {horizon} days ({len(sub)} folds):")
        print(f"    RMSE:    {sub['rmse'].mean():.4f} ± {sub['rmse'].std():.4f}")
        print(f"    MAE:     {sub['mae'].mean():.4f} ± {sub['mae'].std():.4f}")
        print(f"    R²:      {sub['r2'].mean():.4f} ± {sub['r2'].std():.4f}")
        print(f"    Dir Acc: {sub['dir_acc'].mean():.4f} ± {sub['dir_acc'].std():.4f}")
        print(f"    IC:      {sub['spearman_ic'].mean():.4f} ± {sub['spearman_ic'].std():.4f}")
        print(f"    Avg iter: {sub['best_iteration'].mean():.0f}")

    # ── Backtest ──
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = pred_df.sort_values(["horizon", "stock_code", "date"]).drop_duplicates(
        subset=["horizon", "stock_code", "date"], keep="last"
    )
    t0_bt = time.perf_counter()
    trade_records, summary_rows = [], []
    for horizon in horizons:
        for th in TARGET_THRESHOLDS:
            trades, bt = run_backtest(pred_df, horizon, th)
            trade_records.extend(trades)
            fold_sub = metrics_df[metrics_df["horizon"] == horizon]
            summary_rows.append({
                "horizon": horizon, "threshold": th,
                "folds": int(fold_sub["fold"].nunique()),
                "rmse_mean": float(fold_sub["rmse"].mean()),
                "mae_mean": float(fold_sub["mae"].mean()),
                "r2_mean": float(fold_sub["r2"].mean()),
                "dir_acc_mean": float(fold_sub["dir_acc"].mean()),
                "spearman_ic_mean": float(fold_sub["spearman_ic"].mean()),
                **bt,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="cumulative_return", ascending=False, na_position="last"
    ).reset_index(drop=True)

    show_cols = [
        "horizon", "threshold", "total_trades", "win_rate", "cumulative_return",
        "wl_ratio", "avg_holding_days", "max_drawdown",
        "rmse_mean", "mae_mean", "r2_mean", "dir_acc_mean", "spearman_ic_mean",
    ]
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None,
                           "display.width", 250, "display.float_format", "{:.4f}".format):
        print(summary_df[show_cols])

    # ── Save ──
    out_summary = {
        "config": {
            "data": str(csv_path), "initial_train_end": args.initial_train_end,
            "test_window_days": args.test_window_days, "embargo_days": args.embargo_days,
            "horizons": horizons, "n_trials": args.n_trials,
            "tune_budget_sec": args.tune_budget,
            "feature_selection": FEATURE_SELECTION if args.auto_select else "disabled",
            "total_features": len(all_features),
        },
        "tuned_params": {str(h): best_params_per_horizon[h] for h in horizons},
        "fold_metrics": [{k: _to_jsonable(v) for k, v in r.items()} for r in all_fold_metrics],
        "summary": [{k: _to_jsonable(v) for k, v in r.items()} for r in summary_df.to_dict(orient="records")],
    }
    Path(args.out_summary).write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_trades).write_text(
        json.dumps([{k: _to_jsonable(v) for k, v in r.items()} for r in trade_records],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[TIME] backtest: {time.perf_counter() - t0_bt:.2f}s")
    print(f"Saved: {args.out_summary}, {args.out_trades}, {args.out_importance}, {args.out_params}")
    print(f"[TIME] end-to-end: {time.perf_counter() - t0_total:.1f}s")


if __name__ == "__main__":
    main()