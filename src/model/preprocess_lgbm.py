#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LGBM 피처 전처리 v2 — 44개 피처 생성

A. 수익률/상대강도 (10개) — 그대로 사용
B. 거래대금 log→z-score 정규화 (6개)
C. 수급 파생 (12개)
D. 기술적 지표 (8개)
E. RSI 파생 (3개)
F. MACD 파생 (5개)

Input:  주가/수급 CSV + 지수 CSV
Output: features_lgbm.csv (44 features + ID/price cols)
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

BASE_COLS = [
    "종목코드",
    "날짜",
    "종가",
    "최고가",
    "최저가",
    "개인_매수2_거래량",
    "개인_매수2_거래대금",
    "개인_매도_거래량",
    "개인_매도_거래대금",
    "외국인_매수2_거래량",
    "외국인_매수2_거래대금",
    "외국인_매도_거래량",
    "외국인_매도_거래대금",
    "기관계_매수2_거래량",
    "기관계_매수2_거래대금",
    "기관계_매도_거래량",
    "기관계_매도_거래대금",
]

INDEX_FIRST7 = ["지수코드", "지수명", "날짜", "종가", "시가", "고가", "저가"]

# 거래대금 컬럼 (거래량은 제외)
AMT_BUY_COLS = {
    "개인": "개인_매수2_거래대금",
    "외국인": "외국인_매수2_거래대금",
    "기관": "기관계_매수2_거래대금",
}
AMT_SELL_COLS = {
    "개인": "개인_매도_거래대금",
    "외국인": "외국인_매도_거래대금",
    "기관": "기관계_매도_거래대금",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_yyyymmdd_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(raw[missing], errors="coerce")
        dt.loc[missing] = dt2
    return dt


def rolling_zscore(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std"""
    rm = series.rolling(window, min_periods=min_periods).mean()
    rs = series.rolling(window, min_periods=min_periods).std()
    return (series - rm) / rs.replace(0, np.nan)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI (0~100)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram"""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_stock_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "stock_code" in df.columns and "종목코드" not in df.columns:
        df = df.rename(columns={"stock_code": "종목코드"})

    if all(c in df.columns for c in BASE_COLS):
        out = df[BASE_COLS].copy()
    else:
        if df.shape[1] < 17:
            raise ValueError(f"price input has too few columns: {df.shape[1]}")
        out = df.iloc[:, :17].copy()
        out.columns = BASE_COLS

    out["종목코드"] = out["종목코드"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out["날짜"] = parse_yyyymmdd_series(out["날짜"])

    for c in BASE_COLS:
        if c not in ("종목코드", "날짜"):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["종목코드", "날짜", "종가"]).sort_values(["종목코드", "날짜"]).reset_index(drop=True)
    return out


def load_kospi_index(path: Path) -> pd.DataFrame:
    idx = pd.read_csv(path)

    if not all(c in idx.columns for c in INDEX_FIRST7):
        if idx.shape[1] < 7:
            raise ValueError(f"index input has too few columns: {idx.shape[1]}")
        idx = idx.iloc[:, :7].copy()
        idx.columns = INDEX_FIRST7

    idx["지수코드"] = idx["지수코드"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    idx = idx[idx["지수코드"] == "0001"].copy()

    idx["날짜"] = parse_yyyymmdd_series(idx["날짜"])
    idx["종가"] = pd.to_numeric(idx["종가"], errors="coerce")
    idx["고가"] = pd.to_numeric(idx["고가"], errors="coerce")
    idx["저가"] = pd.to_numeric(idx["저가"], errors="coerce")

    idx = idx.dropna(subset=["날짜", "종가"]).sort_values("날짜")
    idx = idx[["날짜", "종가", "고가", "저가"]].drop_duplicates("날짜", keep="last")
    idx = idx.rename(columns={
        "종가": "index_kospi_close",
        "고가": "index_kospi_high",
        "저가": "index_kospi_low",
    })

    idx["index_ret_1"] = idx["index_kospi_close"].pct_change(1)
    idx["index_ret_5"] = idx["index_kospi_close"].pct_change(5)
    idx["index_ret_20"] = idx["index_kospi_close"].pct_change(20)
    return idx


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features(price_df: pd.DataFrame, kospi_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.merge(kospi_df, on="날짜", how="left")
    df = df.sort_values(["종목코드", "날짜"]).reset_index(drop=True)

    g = df.groupby("종목코드", sort=False)

    # ═══════════════════════════════════════════════════════════════════
    # A. 수익률 / 상대강도 (10개)
    # ═══════════════════════════════════════════════════════════════════
    df["ret_1"] = g["종가"].pct_change(1)
    df["ret_5"] = g["종가"].pct_change(5)
    df["ret_20"] = g["종가"].pct_change(20)
    df["ret_1_5_diff"] = df["ret_1"] - df["ret_5"]
    df["ret_5_20_diff"] = df["ret_5"] - df["ret_20"]
    # index_ret_1/5/20 already merged from kospi_df
    df["relative_strength_5"] = df["ret_5"] - df["index_ret_5"]
    df["relative_strength_20"] = df["ret_20"] - df["index_ret_20"]

    # ═══════════════════════════════════════════════════════════════════
    # B. 거래대금 log → z-score 정규화 (6개)
    # ═══════════════════════════════════════════════════════════════════
    for label, buy_col in AMT_BUY_COLS.items():
        log_col = np.log1p(df[buy_col].clip(lower=0))
        df[f"{label}_매수_lz20"] = g.apply(
            lambda x: rolling_zscore(np.log1p(x[buy_col].clip(lower=0)), 20),
        ).droplevel(0) if False else None  # placeholder

    # Vectorized approach (faster than groupby apply)
    for label in ["개인", "외국인", "기관"]:
        buy_col = AMT_BUY_COLS[label]
        sell_col = AMT_SELL_COLS[label]

        # B: log-zscore for buy/sell
        log_buy = np.log1p(df[buy_col].clip(lower=0))
        log_sell = np.log1p(df[sell_col].clip(lower=0))

        df[f"{label}_매수_lz20"] = g.apply(
            lambda x, bc=buy_col: rolling_zscore(np.log1p(x[bc].clip(lower=0)), 20)
        ).reset_index(level=0, drop=True)

        df[f"{label}_매도_lz20"] = g.apply(
            lambda x, sc=sell_col: rolling_zscore(np.log1p(x[sc].clip(lower=0)), 20)
        ).reset_index(level=0, drop=True)

    # ═══════════════════════════════════════════════════════════════════
    # C. 수급 파생 (12개)
    # ═══════════════════════════════════════════════════════════════════

    # 순매수 (log space) per entity
    for label in ["개인", "외국인", "기관"]:
        buy_col = AMT_BUY_COLS[label]
        sell_col = AMT_SELL_COLS[label]
        net_log = np.log1p(df[buy_col].clip(lower=0)) - np.log1p(df[sell_col].clip(lower=0))

        # 순매수 z-score
        df[f"{label}_순매수_lz20"] = g.apply(
            lambda x, bc=buy_col, sc=sell_col: rolling_zscore(
                np.log1p(x[bc].clip(lower=0)) - np.log1p(x[sc].clip(lower=0)), 20
            )
        ).reset_index(level=0, drop=True)

        # 순매수 변화 (diff)
        df[f"{label}_순매수_lz20_diff"] = g[f"{label}_순매수_lz20"].diff(1)

    # 외국인/기관 비중
    total_amt = (
        df["개인_매수2_거래대금"] + df["개인_매도_거래대금"]
        + df["외국인_매수2_거래대금"] + df["외국인_매도_거래대금"]
        + df["기관계_매수2_거래대금"] + df["기관계_매도_거래대금"]
    ).replace(0, np.nan)

    df["외국인_비중"] = (df["외국인_매수2_거래대금"] + df["외국인_매도_거래대금"]) / total_amt
    df["기관_비중"] = (df["기관계_매수2_거래대금"] + df["기관계_매도_거래대금"]) / total_amt

    # 수급 불균형: (전체매수 - 전체매도) / (전체매수 + 전체매도)
    total_buy = df["개인_매수2_거래대금"] + df["외국인_매수2_거래대금"] + df["기관계_매수2_거래대금"]
    total_sell = df["개인_매도_거래대금"] + df["외국인_매도_거래대금"] + df["기관계_매도_거래대금"]
    denom = (total_buy + total_sell).replace(0, np.nan)
    df["수급_불균형"] = (total_buy - total_sell) / denom

    # 외국인/기관 순매수 5일 MA의 z-score
    for label in ["외국인", "기관"]:
        buy_col = AMT_BUY_COLS[label]
        sell_col = AMT_SELL_COLS[label]

        df[f"{label}_순매수_5ma_lz20"] = g.apply(
            lambda x, bc=buy_col, sc=sell_col: rolling_zscore(
                (np.log1p(x[bc].clip(lower=0)) - np.log1p(x[sc].clip(lower=0)))
                .rolling(5, min_periods=3).mean(),
                20
            )
        ).reset_index(level=0, drop=True)

    # 수급 모멘텀: 전체 순매수 5일평균 / 20일평균
    total_net = total_buy - total_sell
    net_5ma = g.apply(lambda x: (
        x["개인_매수2_거래대금"] + x["외국인_매수2_거래대금"] + x["기관계_매수2_거래대금"]
        - x["개인_매도_거래대금"] - x["외국인_매도_거래대금"] - x["기관계_매도_거래대금"]
    ).rolling(5, min_periods=3).mean()).reset_index(level=0, drop=True)

    net_20ma = g.apply(lambda x: (
        x["개인_매수2_거래대금"] + x["외국인_매수2_거래대금"] + x["기관계_매수2_거래대금"]
        - x["개인_매도_거래대금"] - x["외국인_매도_거래대금"] - x["기관계_매도_거래대금"]
    ).rolling(20, min_periods=10).mean()).reset_index(level=0, drop=True)

    df["수급_모멘텀"] = net_5ma / net_20ma.replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════════
    # D. 기술적 지표 (8개)
    # ═══════════════════════════════════════════════════════════════════

    # RSI 14
    df["RSI_14"] = g.apply(lambda x: calc_rsi(x["종가"], 14)).reset_index(level=0, drop=True)

    # Bollinger Band %B (20일)
    ma20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    std20 = g["종가"].transform(lambda x: x.rolling(20, min_periods=15).std())
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_pctB"] = (df["종가"] - bb_lower) / bb_width

    # 이격도 (disparity)
    df["disparity_20"] = df["종가"] / ma20 * 100

    # 변동성 20일
    df["volatility_20"] = g["ret_1"].transform(lambda x: x.rolling(20, min_periods=15).std())

    # 당일 변동폭
    df["high_low_range"] = (df["최고가"] - df["최저가"]) / df["종가"].replace(0, np.nan)

    # MA cross
    ma5 = g["종가"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["ma_cross_5_20"] = ma5 / ma20 - 1
    ma_diff = ma5 - ma20
    df["ma_cross_diff"] = g.apply(
        lambda x: (
            x["종가"].rolling(5, min_periods=3).mean()
            - x["종가"].rolling(20, min_periods=15).mean()
        ).diff(1)
    ).reset_index(level=0, drop=True)

    # RSI diff
    df["rsi_diff"] = g["RSI_14"].diff(1)

    # ═══════════════════════════════════════════════════════════════════
    # E. RSI 파생 (3개)
    # ═══════════════════════════════════════════════════════════════════
    df["rsi_oversold"] = (df["RSI_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["RSI_14"] > 70).astype(int)
    df["rsi_z20"] = g.apply(
        lambda x: rolling_zscore(calc_rsi(x["종가"], 14), 20)
    ).reset_index(level=0, drop=True)

    # ═══════════════════════════════════════════════════════════════════
    # F. MACD 파생 (5개)
    # ═══════════════════════════════════════════════════════════════════
    macd_results = g.apply(lambda x: pd.DataFrame(
        dict(zip(["_macd", "_signal", "_hist"], calc_macd(x["종가"]))),
        index=x.index,
    ))
    if isinstance(macd_results.index, pd.MultiIndex):
        macd_results = macd_results.droplevel(0)

    df["macd_hist"] = macd_results["_hist"]
    df["hist_z20"] = g.apply(
        lambda x: rolling_zscore(
            calc_macd(x["종가"])[2], 20
        )
    ).reset_index(level=0, drop=True)

    df["hist_diff"] = g["macd_hist"].diff(1)

    macd_line = macd_results["_macd"]
    df["macd_slope"] = g.apply(
        lambda x: pd.Series(calc_macd(x["종가"])[0].values, index=x.index).diff(1)
    ).reset_index(level=0, drop=True)

    df["macd_accel"] = g["macd_slope"].diff(1)

    # ═══════════════════════════════════════════════════════════════════
    # Cleanup: add convenience columns
    # ═══════════════════════════════════════════════════════════════════
    df["stock_code"] = df["종목코드"]
    df["date"] = df["날짜"]

    # ─── Select output columns ───
    feature_cols = [
        # A: 수익률/상대강도
        "ret_1", "ret_5", "ret_20", "ret_1_5_diff", "ret_5_20_diff",
        "index_ret_1", "index_ret_5", "index_ret_20",
        "relative_strength_5", "relative_strength_20",
        # B: 거래대금 log-zscore
        "개인_매수_lz20", "개인_매도_lz20",
        "외국인_매수_lz20", "외국인_매도_lz20",
        "기관_매수_lz20", "기관_매도_lz20",
        # C: 수급 파생
        "개인_순매수_lz20", "외국인_순매수_lz20", "기관_순매수_lz20",
        "개인_순매수_lz20_diff", "외국인_순매수_lz20_diff", "기관_순매수_lz20_diff",
        "외국인_비중", "기관_비중", "수급_불균형",
        "외국인_순매수_5ma_lz20", "기관_순매수_5ma_lz20",
        "수급_모멘텀",
        # D: 기술적 지표
        "RSI_14", "BB_pctB", "disparity_20", "volatility_20",
        "high_low_range", "ma_cross_5_20", "ma_cross_diff", "rsi_diff",
        # E: RSI 파생
        "rsi_oversold", "rsi_overbought", "rsi_z20",
        # F: MACD 파생
        "macd_hist", "hist_z20", "hist_diff", "macd_slope", "macd_accel",
    ]

    id_cols = ["종목코드", "날짜", "종가", "최고가", "최저가",
               "index_kospi_close", "index_kospi_high", "index_kospi_low",
               "stock_code", "date"]

    out_cols = id_cols + feature_cols
    out_cols = [c for c in out_cols if c in df.columns]

    return df[out_cols]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LGBM 피처 전처리 v2 (44개 피처)")
    ap.add_argument("--price", default="data/stock/merged_stock.csv", help="주가/수급 CSV 경로")
    ap.add_argument("--index", default="data/index.csv", help="지수 CSV 경로")
    ap.add_argument("--output", default="data/features_lgbm.csv", help="출력 CSV 경로")
    args = ap.parse_args()

    price_path = Path(args.price)
    index_path = Path(args.index)
    out_path = Path(args.output)

    if not price_path.exists():
        raise SystemExit(f"price file not found: {price_path}")
    if not index_path.exists():
        raise SystemExit(f"index file not found: {index_path}")

    print("=" * 60)
    print("LGBM Feature Preprocessing v2 (44 features)")
    print("=" * 60)

    price_df = load_stock_base(price_path)
    print(f"[1/3] Loaded price data: {len(price_df):,} rows, {price_df['종목코드'].nunique()} stocks")

    kospi_df = load_kospi_index(index_path)
    print(f"[2/3] Loaded KOSPI index: {len(kospi_df):,} dates")

    out_df = build_features(price_df, kospi_df)
    print(f"[3/3] Features built: {out_df.shape[1]} columns")

    # NA summary
    feature_cols = [c for c in out_df.columns if c not in [
        "종목코드", "날짜", "종가", "최고가", "최저가",
        "index_kospi_close", "index_kospi_high", "index_kospi_low",
        "stock_code", "date",
    ]]
    na_counts = out_df[feature_cols].isnull().sum()
    na_any = na_counts[na_counts > 0].sort_values(ascending=False)
    if len(na_any) > 0:
        print(f"\n[NA Summary] {len(na_any)} features with NA:")
        for c, v in na_any.items():
            print(f"  {c:<30s} {v:>6,} / {len(out_df):,} ({v / len(out_df) * 100:.1f}%)")
    else:
        print("\n[NA Summary] No NAs in features!")

    total_na_rows = out_df[feature_cols].isnull().any(axis=1).sum()
    print(f"\nRows with any NA: {total_na_rows:,} / {len(out_df):,} ({total_na_rows / len(out_df) * 100:.1f}%)")
    print(f"Clean rows: {len(out_df) - total_na_rows:,} ({(len(out_df) - total_na_rows) / len(out_df) * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Saved: {out_path}")
    print(f"[INFO] rows={len(out_df):,}, stocks={out_df['종목코드'].nunique():,}, features={len(feature_cols)}")


if __name__ == "__main__":
    main()