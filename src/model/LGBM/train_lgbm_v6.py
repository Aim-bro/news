#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM v6 — Peak/Trough Probability Detector.

Two separate models:
  - Peak model:  P(향후 20일 내 최대 하락 >= 10%) → "지금이 고점 근처"
  - Trough model: P(향후 20일 내 최대 상승 >= 10%) → "지금이 저점 근처"

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
from typing import Dict, List

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

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")


# ─── Constants ───────────────────────────────────────────────────────────────

ID_COLS = ["종목코드", "종목명", "날짜", "date", "stock_code"]
PRICE_LEVEL_COLS = [
    "종가", "최고가", "최저가",
    "index_kospi_close", "index_kospi_high", "index_kospi_low",
]

# Peak/Trough definition
HORIZON = 10          # look-ahead window (trading days)
THRESHOLD = 0.10      # 10% move threshold
PROB_THRESHOLDS = [0.50, 0.60, 0.70]  # probability thresholds for backtest

DEFAULT_LGB_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="auc",
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
    is_unbalance=True,
    force_col_wise=True,
    verbosity=-1,
    random_state=42,
    n_jobs=-1,
)

FEATURE_SELECTION = dict(
    warmup_folds=1,
    min_features=10,
    importance_pct=0.90,
)

MODEL_TYPES = ["peak", "trough"]


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

    def update(self, model: lgb.LGBMClassifier, feature_names: List[str],
               fold_id: int, model_type: str):
        imp = model.feature_importances_
        for fname, val in zip(feature_names, imp):
            self.importance_sum[fname] += float(val)
            self.importance_count[fname] += 1
            self.fold_records.append({
                "fold": fold_id, "model_type": model_type,
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


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: np.nan for k in [
            "accuracy", "auc", "f1", "precision", "recall",
            "positive_ratio_true", "positive_ratio_pred",
        ]}
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = np.nan
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    return dict(
        accuracy=acc, auc=auc, f1=f1,
        precision=prec, recall=rec,
        positive_ratio_true=float(y_true.mean()),
        positive_ratio_pred=float(y_pred.mean()),
    )


def build_probability_view(pred_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ["stock_code", "종목코드", "date", "종가"] if c in pred_df.columns]
    base = pred_df[id_cols].drop_duplicates().copy()

    peak = pred_df[pred_df["model_type"] == "peak"][id_cols + ["target", "prob"]].copy()
    peak = peak.rename(columns={"target": "peak_target", "prob": "peak_prob"})

    trough = pred_df[pred_df["model_type"] == "trough"][id_cols + ["target", "prob"]].copy()
    trough = trough.rename(columns={"target": "trough_target", "prob": "trough_prob"})

    out = base.merge(peak, on=id_cols, how="left")
    out = out.merge(trough, on=id_cols, how="left")
    out = out.sort_values([c for c in ["stock_code", "date"] if c in out.columns]).reset_index(drop=True)
    return out


def save_probability_chart(prob_df: pd.DataFrame, stock_code: str, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    work = prob_df[prob_df["stock_code"].astype(str).str.zfill(6) == str(stock_code).zfill(6)].copy()
    if work.empty:
        raise ValueError(f"stock_code not found in probability view: {stock_code}")

    work = work.sort_values("date")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{str(stock_code).zfill(6)}_peak_trough_prob.png"

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(work["date"], work["종가"], color="black", linewidth=1.5, label="Close")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.plot(work["date"], work["peak_prob"], color="#d62728", alpha=0.85, label="Peak Prob")
    ax2.plot(work["date"], work["trough_prob"], color="#1f77b4", alpha=0.85, label="Trough Prob")
    ax2.set_ylabel("Probability", color="gray")
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", labelcolor="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{str(stock_code).zfill(6)} Close vs Peak/Trough Probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ─── Data loading ────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "날짜" in df.columns:
        df["date"] = pd.to_datetime(df["날짜"], errors="coerce")
    else:
        raise ValueError("No date column found.")

    if "stock_code" in df.columns:
        code_src = df["stock_code"]
    elif "종목코드" in df.columns:
        code_src = df["종목코드"]
    else:
        raise ValueError("No stock code column found.")
    df["stock_code"] = code_src.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(6)
    if "종목코드" not in df.columns:
        df["종목코드"] = df["stock_code"]

    skip = set(ID_COLS) | {"date"}
    for c in df.columns:
        if c not in skip and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    na_before = int(df.isnull().sum().sum())
    print(f"  NA in data: {na_before:,} cells (LGBM handles natively)")

    return df


def add_targets(df: pd.DataFrame, horizon: int = 20, threshold: float = 0.10) -> pd.DataFrame:
    """
    Create peak/trough binary targets.

    Peak target (target_peak):
      향후 horizon일 내 최저가 / 오늘 종가 - 1 <= -threshold → 1
      "오늘 이후 큰 하락이 온다" = 지금이 고점 근처

    Trough target (target_trough):
      향후 horizon일 내 최고가 / 오늘 종가 - 1 >= +threshold → 1
      "오늘 이후 큰 상승이 온다" = 지금이 저점 근처
    """
    out = df.copy().sort_values(["stock_code", "date"]).reset_index(drop=True)
    g = out.groupby("stock_code", sort=False)

    # Rolling future min/max price over horizon window
    # shift(-1) so we don't include today, then rolling min/max over horizon
    future_min = g["종가"].transform(
        lambda x: x.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    )
    future_max = g["종가"].transform(
        lambda x: x.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )

    # Max drawdown from today = (future_min / today_price) - 1
    max_drawdown = future_min / out["종가"] - 1.0
    # Max rally from today = (future_max / today_price) - 1
    max_rally = future_max / out["종가"] - 1.0

    # Peak: big drop coming → 1
    out["target_peak"] = (max_drawdown <= -threshold).astype(float)
    out.loc[future_min.isna(), "target_peak"] = np.nan

    # Trough: big rally coming → 1
    out["target_trough"] = (max_rally >= threshold).astype(float)
    out.loc[future_max.isna(), "target_trough"] = np.nan

    # Store raw values for analysis
    out["fwd_max_drawdown"] = max_drawdown
    out["fwd_max_rally"] = max_rally

    return out


def build_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [c for c in df.columns if c.startswith("fwd_") or c.startswith("target_")]
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

def _make_inner_splits(train_df, n_splits=3, embargo_days=0):
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
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "is_unbalance": True,
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

    auc_scores = []
    for tr_idx, va_idx in inner_splits:
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target_col]
        X_va = train_df.iloc[va_idx][features]
        y_va = train_df.iloc[va_idx][target_col]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        y_prob = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        try:
            auc = roc_auc_score(y_va.values, y_prob)
            auc_scores.append(float(auc))
        except ValueError:
            pass

    return float(np.mean(auc_scores)) if auc_scores else 0.5


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

    print(f"    [Optuna] {len(study.trials)} trials | Best AUC: {study.best_value:.4f}")
    print(f"    [Optuna] Best params: {json.dumps(study.best_params, indent=2)}")

    bp = study.best_params
    best_params = {
        "objective": "binary", "boosting_type": "gbdt", "metric": "auc",
        "is_unbalance": True,
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
               tracker, fold_id, model_type):
    tr_mask, va_mask = _train_valid_split(fold_train)
    train_part = fold_train.iloc[np.where(tr_mask)[0]]
    valid_part = fold_train.iloc[np.where(va_mask)[0]]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        train_part[features], train_part[target_col],
        eval_set=[(valid_part[features], valid_part[target_col])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    tracker.update(model, features, fold_id, model_type)
    y_prob = model.predict_proba(fold_test[features], num_iteration=model.best_iteration_)[:, 1]
    return y_prob, model


# ─── Backtest (peak/trough signal based) ─────────────────────────────────────

def run_backtest(pred_df, model_type, prob_threshold):
    """
    Backtest using peak/trough probability.

    Peak model (매도 시그널):
      - peak_prob > threshold → 매도/숏 진입
      - peak_prob < 0.3 → 청산

    Trough model (매수 시그널):
      - trough_prob > threshold → 매수 진입
      - trough_prob < 0.3 → 청산
    """
    work = pred_df[pred_df["model_type"] == model_type].copy()
    work = work.sort_values(["stock_code", "date"]).reset_index(drop=True)
    trades = []

    is_buy = (model_type == "trough")  # trough = buy signal, peak = sell signal

    for stock_code, g_df in work.groupby("stock_code", sort=False):
        pos = None
        for _, row in g_df.iterrows():
            px = float(row["종가"])
            prob = float(row["prob"])
            dt = pd.Timestamp(row["date"])

            if pos is None:
                if prob > prob_threshold:
                    pos = {
                        "stock_code": stock_code,
                        "entry_date": dt, "entry_price": px,
                        "entry_prob": prob, "bars": 0,
                        "direction": "long" if is_buy else "short",
                    }
                continue

            pos["bars"] += 1
            if is_buy:
                ret_now = px / pos["entry_price"] - 1.0
            else:
                ret_now = pos["entry_price"] / px - 1.0  # short return

            exit_by_signal = prob < 0.3           # signal weakened
            exit_by_time = pos["bars"] > 40       # max 40 bars (~2 months)
            exit_by_stop = ret_now <= -0.15       # stop loss

            if exit_by_signal or exit_by_time or exit_by_stop:
                trades.append({
                    "model_type": model_type, "prob_threshold": prob_threshold,
                    "stock_code": pos["stock_code"],
                    "direction": pos["direction"],
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "entry_price": pos["entry_price"], "exit_price": px,
                    "return": ret_now, "holding_days": pos["bars"],
                    "exit_reason": "signal" if exit_by_signal else ("stop" if exit_by_stop else "time"),
                    "entry_prob": pos["entry_prob"], "exit_prob": prob,
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
    ap = argparse.ArgumentParser(description="LightGBM v6 — Peak/Trough Probability")
    ap.add_argument("--data", default="data/features_lgbm.csv")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
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
    ap.add_argument("--out-probs", default="probability_view.csv")
    ap.add_argument("--plot-stock", default=None, help="Save price/probability chart for one stock code")
    ap.add_argument("--plot-dir", default="plots")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    print("=" * 60)
    print("LightGBM v6 — Peak/Trough Probability Detector")
    print(f"  Horizon: {args.horizon} days, Threshold: {args.threshold*100:.0f}%")
    print("=" * 60)
    print(f"Data: {csv_path}")

    t0_load = time.perf_counter()
    df = load_and_prepare(csv_path)
    df = add_targets(df, horizon=args.horizon, threshold=args.threshold)
    all_features = build_features(df)
    t1_load = time.perf_counter()

    print(f"Rows: {len(df):,}, Stocks: {df['stock_code'].nunique():,}")
    print(f"Features ({len(all_features)}): {all_features}")
    print(f"Embargo: {args.embargo_days} days | Auto-select: {args.auto_select}")
    print(f"Optuna: {args.n_trials} trials/model, {args.tune_budget}s budget")

    # Class balance
    for mt in MODEL_TYPES:
        tc = f"target_{mt}"
        valid = df[tc].dropna()
        pos_pct = valid.mean() * 100
        print(f"  {mt.upper()} class balance: {pos_pct:.1f}% positive / "
              f"{100-pos_pct:.1f}% negative ({len(valid):,} samples)")
    print(f"[TIME] data prep: {t1_load - t0_load:.2f}s\n")

    folds = make_folds(df, args.initial_train_end, args.test_window_days, args.embargo_days)
    if not folds:
        raise SystemExit("No folds generated.")
    print(f"Folds: {len(folds)}")
    for f in folds:
        print(f"  F{f.fold_id}: train ≤ {f.train_end_date.date()} | "
              f"test {f.test_start_date.date()} ~ {f.test_end_date.date()}")

    # ── Hyperparameter tuning ──
    best_params_per_model: Dict[str, Dict] = {}

    if args.load_params:
        loaded = json.loads(Path(args.load_params).read_text(encoding="utf-8"))
        for mt in MODEL_TYPES:
            best_params_per_model[mt] = loaded.get(mt, DEFAULT_LGB_PARAMS.copy())
            print(f"\n[{mt.upper()}] Loaded params from {args.load_params}")
    elif args.skip_tune:
        for mt in MODEL_TYPES:
            best_params_per_model[mt] = DEFAULT_LGB_PARAMS.copy()
        print("\nSkipping Optuna, using defaults")
    else:
        print(f"\n{'='*60}")
        print("Phase 1: Optuna Hyperparameter Tuning (maximize AUC)")
        print(f"{'='*60}")

        for mt in MODEL_TYPES:
            t0_tune = time.perf_counter()
            target_col = f"target_{mt}"
            print(f"\n--- Model: {mt.upper()} ---")

            tune_end = folds[0].train_end_date
            cols_needed = ["stock_code", "date", "종가", target_col] + all_features
            cols_needed = [c for c in cols_needed if c in df.columns]
            ds = df[cols_needed].copy()
            mask = ds[target_col].notna()
            tune_data = ds.loc[mask & (ds["date"] <= tune_end)].reset_index(drop=True)

            pos_pct = tune_data[target_col].mean() * 100
            print(f"  Tune data: {len(tune_data):,} rows (≤ {tune_end.date()}), "
                  f"class balance: {pos_pct:.1f}% positive")

            if len(tune_data) < 500:
                print("  Too few rows, using defaults")
                best_params_per_model[mt] = DEFAULT_LGB_PARAMS.copy()
            else:
                best_params_per_model[mt] = run_optuna_tuning(
                    tune_data, all_features, target_col,
                    n_trials=args.n_trials, time_budget_sec=args.tune_budget,
                    embargo_days=args.embargo_days,
                )
            print(f"  [TIME] tune {mt}: {time.perf_counter() - t0_tune:.1f}s")

        Path(args.out_params).write_text(
            json.dumps(best_params_per_model, indent=2), encoding="utf-8",
        )
        print(f"\nSaved: {args.out_params}")

    # ── Walk-forward evaluation ──
    print(f"\n{'='*60}")
    print("Phase 2: Walk-Forward Evaluation")
    print(f"{'='*60}")

    all_fold_metrics, all_preds = [], []

    t0_train = time.perf_counter()
    for mt in MODEL_TYPES:
        t0_m = time.perf_counter()
        target_col = f"target_{mt}"
        lgb_params = best_params_per_model[mt]
        tracker = FeatureTracker()

        print(f"\n{'─'*50}")
        print(f"Model: {mt.upper()} (horizon={args.horizon}d, threshold={args.threshold*100:.0f}%)")
        print(f"  lr={lgb_params.get('learning_rate', '?'):.4f}, "
              f"n_est={lgb_params.get('n_estimators', '?')}, "
              f"depth={lgb_params.get('max_depth', '?')}, "
              f"leaves={lgb_params.get('num_leaves', '?')}")
        print(f"{'─'*50}")

        cols_needed = ["stock_code", "종목코드", "date", "종가",
                       target_col, "fwd_max_drawdown", "fwd_max_rally"] + all_features
        cols_needed = list(dict.fromkeys([c for c in cols_needed if c in df.columns]))
        ds = df[cols_needed].copy()
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

            pos_tr = tr[target_col].mean() * 100
            pos_te = te[target_col].mean() * 100

            # Auto feature selection
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

            print(f"  [{mt.upper()}][F{fold.fold_id}] train={len(tr):,} (pos:{pos_tr:.0f}%) "
                  f"test={len(te):,} (pos:{pos_te:.0f}%) feats={len(active_features)}")

            y_prob, model = train_fold(
                tr, te, target_col, active_features,
                lgb_params, tracker, fold.fold_id, mt,
            )
            y_true = te[target_col].values.astype(int)
            m = calc_metrics(y_true, y_prob, threshold=0.5)

            all_fold_metrics.append({
                "model_type": mt, "fold": fold.fold_id,
                "train_end": fold.train_end_date, "test_start": fold.test_start_date,
                "test_end": fold.test_end_date, "n_features": len(active_features),
                "best_iteration": model.best_iteration_, **m, "rows_test": int(len(te)),
            })
            print(f"    Acc={m['accuracy']:.3f} AUC={m['auc']:.3f} F1={m['f1']:.3f} "
                  f"Prec={m['precision']:.3f} Rec={m['recall']:.3f} "
                  f"(iter={model.best_iteration_})")

            pred_rows = te[["stock_code", "종목코드", "date", "종가"]].copy()
            pred_rows["model_type"] = mt
            pred_rows["target"] = y_true
            pred_rows["prob"] = y_prob
            all_preds.append(pred_rows)

        # Feature importance per model
        imp_summary = tracker.summary(all_features)
        if len(imp_summary) > 0:
            print(f"\n  [{mt.upper()}] Feature Importance (top 15):")
            for i, row in imp_summary.head(15).iterrows():
                cat = "[FLOW]" if any(k in row["feature"] for k in ["외국인", "기관", "개인", "수급"]) else "[TECH]"
                print(f"    {i+1:3d}. {cat} {row['feature']:<35s} imp={row['avg_importance']:.1f}")

            imp_df = tracker.to_dataframe()
            if not imp_df.empty:
                imp_df.to_csv(
                    args.out_importance.replace(".csv", f"_{mt}.csv"),
                    index=False, encoding="utf-8-sig",
                )

        print(f"  [TIME] {mt}: {time.perf_counter() - t0_m:.2f}s")

    print(f"\n[TIME] evaluation: {time.perf_counter() - t0_train:.2f}s")

    if not all_fold_metrics:
        raise SystemExit("No predictions produced.")

    # ── Prediction metrics ──
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(f"\n{'='*60}")
    print("Peak/Trough Detection Performance (mean ± std)")
    print(f"{'='*60}")
    for mt in MODEL_TYPES:
        sub = metrics_df[metrics_df["model_type"] == mt]
        if len(sub) == 0:
            continue
        label = "🔴 PEAK (고점)" if mt == "peak" else "🟢 TROUGH (저점)"
        print(f"\n  {label} ({len(sub)} folds):")
        print(f"    Accuracy:  {sub['accuracy'].mean():.4f} ± {sub['accuracy'].std():.4f}")
        print(f"    AUC:       {sub['auc'].mean():.4f} ± {sub['auc'].std():.4f}")
        print(f"    F1:        {sub['f1'].mean():.4f} ± {sub['f1'].std():.4f}")
        print(f"    Precision: {sub['precision'].mean():.4f} ± {sub['precision'].std():.4f}")
        print(f"    Recall:    {sub['recall'].mean():.4f} ± {sub['recall'].std():.4f}")
        print(f"    Avg iter:  {sub['best_iteration'].mean():.0f}")

    # ── Backtest ──
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = pred_df.sort_values(["model_type", "stock_code", "date"]).drop_duplicates(
        subset=["model_type", "stock_code", "date"], keep="last"
    )
    prob_view_df = build_probability_view(pred_df)
    prob_view_df.to_csv(args.out_probs, index=False, encoding="utf-8-sig")
    print(f"Saved probability view: {args.out_probs}")
    if args.plot_stock:
        chart_path = save_probability_chart(prob_view_df, args.plot_stock, Path(args.plot_dir))
        print(f"Saved probability chart: {chart_path}")

    t0_bt = time.perf_counter()
    trade_records, summary_rows = [], []
    for mt in MODEL_TYPES:
        for th in PROB_THRESHOLDS:
            trades, bt = run_backtest(pred_df, mt, th)
            trade_records.extend(trades)
            fold_sub = metrics_df[metrics_df["model_type"] == mt]
            summary_rows.append({
                "model_type": mt, "prob_threshold": th,
                "folds": int(fold_sub["fold"].nunique()),
                "accuracy_mean": float(fold_sub["accuracy"].mean()),
                "auc_mean": float(fold_sub["auc"].mean()),
                "f1_mean": float(fold_sub["f1"].mean()),
                **bt,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="win_rate", ascending=False, na_position="last"
    ).reset_index(drop=True)

    show_cols = [
        "model_type", "prob_threshold",
        "total_trades", "win_rate", "cumulative_return", "wl_ratio",
        "avg_holding_days", "max_drawdown",
        "accuracy_mean", "auc_mean", "f1_mean",
    ]
    print(f"\n{'='*60}")
    print("Backtest Results (sorted by win_rate)")
    print(f"{'='*60}")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None,
                           "display.width", 250, "display.float_format", "{:.4f}".format):
        print(summary_df[show_cols])

    # ── Save ──
    out_summary = {
        "config": {
            "data": str(csv_path), "initial_train_end": args.initial_train_end,
            "test_window_days": args.test_window_days, "embargo_days": args.embargo_days,
            "horizon": args.horizon, "threshold": args.threshold,
            "model_types": MODEL_TYPES, "model_type": "peak_trough_binary",
            "n_trials": args.n_trials, "tune_budget_sec": args.tune_budget,
            "feature_selection": FEATURE_SELECTION if args.auto_select else "disabled",
            "total_features": len(all_features),
            "prob_thresholds": PROB_THRESHOLDS,
        },
        "tuned_params": best_params_per_model,
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
