#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Two-stage LightGBM regression for N-day forward returns (Korean stocks).

Single-file pipeline:
- load ./data/features_with_news.csv
- build forward returns (5/10/20)
- expanding walk-forward validation
- train 3 versions (A/B/C)
- evaluate + simple signal backtest
- save trades.json / summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


ID_COLS = ["종목코드", "종목명", "날짜", "date", "stock_code"]
PRICE_LEVEL_COLS = [
    "종가", "최고가", "최저가",
    "index_kospi_close", "index_kospi_high", "index_kospi_low",
    "index_close", "index_high", "index_low",
]
NEWS_COLS = [
    "news_count",
    "sent_mean", "sent_pos_ratio", "sent_neg_ratio",
    "sent_max", "sent_min", "sent_3ma", "sent_momentum",
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


@dataclass
class Fold:
    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray


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

    # 뉴스 컬럼은 2025-01-15 이후부터 존재하므로 결측을 0으로 채움
    for c in NEWS_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["has_news"] = (df["news_count"] > 0).astype(int)

    # Numeric coercion (keep id/date/name untouched)
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


def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [c for c in df.columns if c.startswith("fwd_")]
    exclude_common = set(ID_COLS) | {"date"} | set(target_cols) | set(PRICE_LEVEL_COLS)

    stage1 = [
        c for c in numeric_cols
        if c not in exclude_common and c not in set(NEWS_COLS) and c != "has_news"
    ]
    stage2 = [c for c in numeric_cols if c not in exclude_common]

    if not stage1:
        raise ValueError("No Stage1 features found.")
    if not stage2:
        raise ValueError("No Stage2 features found.")

    return stage1, stage2


def make_folds(
    df: pd.DataFrame,
    initial_train_end: str,
    test_window_days: int,
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
        test_start_idx = train_end_idx + 1
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_start_idx + test_window_days - 1, len(dates) - 1)
        train_end = pd.Timestamp(dates[train_end_idx])
        test_start = pd.Timestamp(dates[test_start_idx])
        test_end = pd.Timestamp(dates[test_end_idx])

        train_mask = (df["date"] <= train_end).values
        test_mask = ((df["date"] >= test_start) & (df["date"] <= test_end)).values
        if test_mask.sum() == 0:
            break

        folds.append(Fold(
            fold_id=fold_id,
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=test_end,
            train_mask=train_mask,
            test_mask=test_mask,
        ))
        fold_id += 1
        train_end_idx = test_end_idx
        if train_end_idx >= len(dates) - 2:
            break

    return folds


def _train_valid_split(train_df: pd.DataFrame, min_valid_dates: int = 20) -> Tuple[np.ndarray, np.ndarray]:
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
        # fallback
        cut = max(1, int(len(train_df) * 0.8))
        tr_mask = np.zeros(len(train_df), dtype=bool)
        tr_mask[:cut] = True
        va_mask = ~tr_mask
    return tr_mask, va_mask


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_model=None,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    callbacks = [lgb.early_stopping(30, verbose=False)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=callbacks,
        init_model=init_model,
    )
    return model


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {k: np.nan for k in ["rmse", "mae", "r2", "dir_acc", "spearman_ic"]}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    ic = _safe_spearman(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc, spearman_ic=ic)


def train_fold_versions(
    fold_train: pd.DataFrame,
    fold_test: pd.DataFrame,
    target_col: str,
    stage1_feats: List[str],
    stage2_feats: List[str],
) -> Dict[str, np.ndarray]:
    tr_mask, va_mask = _train_valid_split(fold_train)
    train_part = fold_train.iloc[np.where(tr_mask)[0]]
    valid_part = fold_train.iloc[np.where(va_mask)[0]]

    preds: Dict[str, np.ndarray] = {}

    # A) No news model (true Stage1 feature set)
    m_a = fit_lgbm(
        train_part[stage1_feats], train_part[target_col],
        valid_part[stage1_feats], valid_part[target_col],
    )
    preds["A_no_news"] = m_a.predict(fold_test[stage1_feats], num_iteration=m_a.best_iteration_)

    # B) News included directly
    m_b = fit_lgbm(
        train_part[stage2_feats], train_part[target_col],
        valid_part[stage2_feats], valid_part[target_col],
    )
    preds["B_news_direct"] = m_b.predict(fold_test[stage2_feats], num_iteration=m_b.best_iteration_)

    # C) Two-stage fine-tuned model (same schema required for init_model)
    # Stage1 for C uses stage2 feature schema but zeroed news columns => no news information leakage/use.
    train_part_c = train_part.copy()
    valid_part_c = valid_part.copy()
    test_part_c = fold_test.copy()
    for c in NEWS_COLS + ["has_news"]:
        if c in train_part_c.columns:
            train_part_c[c] = 0.0
            valid_part_c[c] = 0.0
            test_part_c[c] = 0.0

    m_c_stage1 = fit_lgbm(
        train_part_c[stage2_feats], train_part_c[target_col],
        valid_part_c[stage2_feats], valid_part_c[target_col],
    )
    base_pred_c = m_c_stage1.predict(test_part_c[stage2_feats], num_iteration=m_c_stage1.best_iteration_)

    # Stage2 fine-tune only on news rows (>0) with actual news features, continuing from stage1 booster.
    news_train = train_part["news_count"] > 0
    news_valid = valid_part["news_count"] > 0
    news_test = fold_test["news_count"] > 0

    pred_c = base_pred_c.copy()
    min_rows = 50
    if news_train.sum() >= min_rows and news_valid.sum() >= max(10, min_rows // 5):
        m_c_stage2 = fit_lgbm(
            train_part.loc[news_train, stage2_feats], train_part.loc[news_train, target_col],
            valid_part.loc[news_valid, stage2_feats], valid_part.loc[news_valid, target_col],
            init_model=m_c_stage1.booster_,
        )
        if news_test.any():
            pred_c_news = m_c_stage2.predict(
                fold_test.loc[news_test, stage2_feats],
                num_iteration=m_c_stage2.best_iteration_,
            )
            pred_c[np.where(news_test.values)[0]] = pred_c_news
    preds["C_two_stage"] = pred_c

    return preds


def run_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    version: str,
    threshold: float,
) -> Tuple[List[dict], Dict[str, float]]:
    work = pred_df[(pred_df["horizon"] == horizon) & (pred_df["version"] == version)].copy()
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
                    "version": version,
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


def main():
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser(description="Two-stage LightGBM forward-return pipeline (walk-forward)")
    ap.add_argument("--data", default="data/features_with_news.csv", help="Input CSV path")
    ap.add_argument("--initial-train-end", default="2024-06-30", help="Initial train end date (YYYY-MM-DD)")
    ap.add_argument("--test-window-days", type=int, default=60, help="Walk-forward test window size in trading days")
    ap.add_argument("--out-summary", default="summary.json", help="Summary JSON output")
    ap.add_argument("--out-trades", default="trades.json", help="Trades JSON output")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    horizons = [5, 10, 20]
    print(f"Loading: {csv_path}")
    t0_load = time.perf_counter()
    df = load_and_prepare(csv_path)
    df = add_targets(df, horizons)
    stage1_feats, stage2_feats = build_feature_sets(df)
    t1_load = time.perf_counter()
    print(f"Rows: {len(df):,}, Stocks: {df['stock_code'].nunique():,}")
    print(f"Stage1 features: {len(stage1_feats)} | Stage2 features: {len(stage2_feats)}")
    print(f"[TIME] data prep: {t1_load - t0_load:.2f}s")

    folds = make_folds(df, args.initial_train_end, args.test_window_days)
    if not folds:
        raise SystemExit("No walk-forward folds generated. Check date range / initial-train-end.")
    print(f"Folds: {len(folds)}")

    all_fold_metrics: List[dict] = []
    all_preds: List[pd.DataFrame] = []

    t0_train = time.perf_counter()
    for horizon in horizons:
        t0_h = time.perf_counter()
        target_col = f"fwd_{horizon}"
        base_cols = ["stock_code", "종목명", "date", "종가"] + [target_col] + stage2_feats
        base_cols = list(dict.fromkeys([c for c in base_cols if c in df.columns]))
        ds = df[base_cols].copy()

        # Version-wise NA drop counts (feature cols + target)
        mask_a = ds[stage1_feats + [target_col]].notna().all(axis=1)
        mask_b = ds[stage2_feats + [target_col]].notna().all(axis=1)
        print(f"[H{horizon}] rows removed (A no-news features): {int((~mask_a).sum()):,}")
        print(f"[H{horizon}] rows removed (B/C stage2 features): {int((~mask_b).sum()):,}")

        # Use superset-clean dataset for training all versions to keep fold alignment stable.
        ds_clean = ds.loc[mask_b].copy().reset_index(drop=True)
        if len(ds_clean) == 0:
            print(f"[H{horizon}] skipped: no rows after NA drop")
            continue

        for fold in folds:
            tr = ds_clean.loc[ds_clean["date"] <= fold.train_end_date].copy()
            te = ds_clean.loc[(ds_clean["date"] >= fold.test_start_date) & (ds_clean["date"] <= fold.test_end_date)].copy()
            if len(tr) < 500 or len(te) == 0:
                continue

            preds = train_fold_versions(tr, te, target_col, stage1_feats, stage2_feats)
            y_true = te[target_col].values

            for version, y_pred in preds.items():
                m = calc_metrics(y_true, y_pred)
                rec = {
                    "horizon": horizon,
                    "version": version,
                    "fold": fold.fold_id,
                    "train_end": fold.train_end_date,
                    "test_start": fold.test_start_date,
                    "test_end": fold.test_end_date,
                    **m,
                    "rows_test": int(len(te)),
                }
                all_fold_metrics.append(rec)
                print(
                    f"[H{horizon}][{version}][F{fold.fold_id}] "
                    f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f} "
                    f"Dir={m['dir_acc']:.3f} IC={m['spearman_ic']:.3f}"
                )

                pred_rows = te[["stock_code", "종목명", "date", "종가"]].copy()
                pred_rows["horizon"] = horizon
                pred_rows["version"] = version
                pred_rows["target"] = y_true
                pred_rows["pred"] = y_pred
                all_preds.append(pred_rows)
        t1_h = time.perf_counter()
        print(f"[TIME] horizon {horizon}: {t1_h - t0_h:.2f}s")
    t1_train = time.perf_counter()
    print(f"[TIME] model train/eval total: {t1_train - t0_train:.2f}s")

    if not all_fold_metrics or not all_preds:
        raise SystemExit("No predictions produced. Check data availability and fold settings.")

    metrics_df = pd.DataFrame(all_fold_metrics)
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = pred_df.sort_values(["version", "horizon", "stock_code", "date"]).drop_duplicates(
        subset=["version", "horizon", "stock_code", "date"], keep="last"
    )

    # Backtest all configs
    t0_bt = time.perf_counter()
    trade_records: List[dict] = []
    summary_rows: List[dict] = []
    for horizon in horizons:
        for version in ["A_no_news", "B_news_direct", "C_two_stage"]:
            for th in TARGET_THRESHOLDS:
                trades, bt = run_backtest(pred_df, horizon, version, th)
                trade_records.extend(trades)

                fold_sub = metrics_df[(metrics_df["horizon"] == horizon) & (metrics_df["version"] == version)]
                summary_rows.append({
                    "version": version,
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
    print(f"[TIME] backtest total: {t1_bt - t0_bt:.2f}s")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="cumulative_return", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # Print summary table
    show_cols = [
        "version", "horizon", "threshold",
        "total_trades", "win_rate", "cumulative_return", "wl_ratio",
        "avg_holding_days", "max_drawdown",
        "rmse_mean", "mae_mean", "r2_mean", "dir_acc_mean", "spearman_ic_mean",
    ]
    print("\n=== Ranked Configurations (by cumulative_return) ===")
    with pd.option_context("display.max_rows", 200, "display.width", 220, "display.float_format", "{:.4f}".format):
        print(summary_df[show_cols])

    out_summary = {
        "config": {
            "data": str(csv_path),
            "initial_train_end": args.initial_train_end,
            "test_window_days": args.test_window_days,
            "horizons": horizons,
            "thresholds": TARGET_THRESHOLDS,
            "lgb_params": LGB_PARAMS,
            "stage1_feature_count": len(stage1_feats),
            "stage2_feature_count": len(stage2_feats),
            "note_two_stage": "Stage1 for version C is trained on stage2 schema with news features zeroed to keep LightGBM init_model feature dimensions consistent.",
        },
        "fold_metrics": [{k: _to_jsonable(v) for k, v in r.items()} for r in all_fold_metrics],
        "summary": [{k: _to_jsonable(v) for k, v in r.items()} for r in summary_df.to_dict(orient="records")],
    }
    out_trades = [{k: _to_jsonable(v) for k, v in r.items()} for r in trade_records]

    Path(args.out_summary).write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_trades).write_text(json.dumps(out_trades, ensure_ascii=False, indent=2), encoding="utf-8")
    t1_total = time.perf_counter()
    print(f"\nSaved: {args.out_summary}, {args.out_trades}")
    print(f"[TIME] end-to-end total: {t1_total - t0_total:.2f}s")


if __name__ == "__main__":
    main()
