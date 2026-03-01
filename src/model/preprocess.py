"""
=============================================================
preprocess.py — Raw Data → features_with_news.csv
=============================================================
Converts raw price/supply CSV + raw news CSV into the
features file used by the model.

Pipeline:
  Step 1: Load raw price/supply data (23stocks.csv)
  Step 2: Add stock name mapping
  Step 3: Compute technical indicators (RSI, MACD, MA, BB, etc.)
  Step 4: Compute supply/demand features (net buying, momentum)
  Step 5: Clean news data (df_filtered.csv)
  Step 6: Sentiment analysis (Korean financial keyword dict)
  Step 7: Aggregate daily sentiment per stock
  Step 8: Merge price features + news sentiment
  Step 9: Save features_with_news.csv

Usage:
  python preprocess.py
  python preprocess.py --price 23stocks.csv --news df_filtered.csv
  python preprocess.py --price 23stocks.csv  # no news, skip sentiment

Requires: pandas, numpy
=============================================================
"""

import pandas as pd
import numpy as np
import argparse
import re
import warnings
warnings.filterwarnings("ignore")

# Local defaults for this news/ project layout.
# Update the two input paths to your actual incoming raw files.
DEFAULT_PRICE_INPUT = "data/stock/merged_stock.csv"
DEFAULT_NEWS_INPUT = "data/stock/all_news.csv"  # or pass --news none
DEFAULT_INDEX_INPUT = "data/index.csv"
DEFAULT_FEATURES_OUTPUT = "data/features_with_news.csv"


# ─── Stock Code → Name Mapping ───
STOCK_NAMES = {
    5380: "현대차", 5490: "POSCO홀딩스", 6400: "삼성SDI",
    6800: "미래에셋증권", 12330: "현대모비스", 12450: "한화에어로스페이스",
    32830: "삼성생명", 34020: "두산에너빌리티", 35420: "네이버",
    35720: "카카오", 42670: "JYP엔터", 36570: "NCsoft",
    39490: "키움증권", 42660: "한화오션", 52380: "한미반도체",
    71050: "한국금융지주", 247540: "에코프로", 247560: "에코프로비엠",
    259960: "크래프톤", 42700: "현대두산인프라코어", 298040: "효성첨단소재",
    352820: "하이브", 377300: "카카오페이",
}


# ═══════════════════════════════════════════════════════════
# STEP 1-2: Load & Map
# ═══════════════════════════════════════════════════════════

# Runtime-normalize stock name mapping to 6-digit stock codes (e.g., 005380)
STOCK_NAMES = {f"{int(k):06d}": v for k, v in STOCK_NAMES.items()}
STOCK_NAMES_6 = STOCK_NAMES


def load_price_data(path):
    """Load raw price/supply CSV."""
    df = pd.read_csv(path)

    # Normalize column names
    if "stock_code" in df.columns and "종목코드" not in df.columns:
        df = df.rename(columns={"stock_code": "종목코드"})

    # Standardize to 6-digit stock code strings (e.g., 5380 -> 005380)
    df["종목코드"] = df["종목코드"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    # Parse date (handle both 20230102 and 2023-01-02 formats)
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str), format="mixed")

    # Add stock names
    if "종목명" not in df.columns:
        df["종목명"] = df["종목코드"].map(STOCK_NAMES_6)
        df = df.dropna(subset=["종목명"])

    df = df.sort_values(["종목코드", "날짜"]).reset_index(drop=True)
    print(f"  Price data: {len(df):,} rows, {df['종목명'].nunique()} stocks, "
          f"{df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 3: Technical Indicators
# ═══════════════════════════════════════════════════════════

def add_technical_indicators(df):
    """Add RSI, MACD, MA, Bollinger Bands, ATR, Disparity per stock."""
    results = []

    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        close = g["종가"]
        high = g["최고가"]
        low = g["최저가"]

        # RSI (14-day)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]

        # Moving Averages
        g["MA_5"] = close.rolling(5).mean()
        g["MA_20"] = close.rolling(20).mean()
        g["MA_60"] = close.rolling(60).mean()

        # Disparity (price vs MA)
        g["이격도_5"] = (close / g["MA_5"] * 100).round(2)
        g["이격도_20"] = (close / g["MA_20"] * 100).round(2)

        # MA Cross (1 = golden, -1 = dead, 0 = none)
        g["MA_Cross"] = 0
        g.loc[(g["MA_5"] > g["MA_20"]) & (g["MA_5"].shift(1) <= g["MA_20"].shift(1)), "MA_Cross"] = 1
        g.loc[(g["MA_5"] < g["MA_20"]) & (g["MA_5"].shift(1) >= g["MA_20"].shift(1)), "MA_Cross"] = -1

        # Bollinger Bands (20-day)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        g["BB_Upper"] = ma20 + 2 * std20
        g["BB_Lower"] = ma20 - 2 * std20
        bb_range = (g["BB_Upper"] - g["BB_Lower"]).replace(0, np.nan)
        g["BB_PctB"] = (close - g["BB_Lower"]) / bb_range

        # ATR (14-day)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()
        g["ATR_Ratio"] = (g["ATR_14"] / close * 100).round(2)

        results.append(g)

    df = pd.concat(results, ignore_index=True)
    print(f"  Technical indicators added: RSI, MACD, MA(5/20/60), BB, ATR, Disparity")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 4: Supply/Demand Features
# ═══════════════════════════════════════════════════════════

def add_supply_features(df):
    """Compute net buying, moving averages, momentum for each investor group."""
    results = []

    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()

        for prefix, buy_col, sell_col in [
            ("개인", "개인_매수2_거래대금", "개인_매도_거래대금"),
            ("외국인", "외국인_매수2_거래대금", "외국인_매도_거래대금"),
            ("기관계", "기관계_매수2_거래대금", "기관계_매도_거래대금"),
        ]:
            net = g[buy_col].fillna(0) - g[sell_col].fillna(0)
            g[f"{prefix}_순매수"] = net
            g[f"{prefix}_순매수_5MA"] = net.rolling(5).mean()
            g[f"{prefix}_순매수_10MA"] = net.rolling(10).mean()
            g[f"{prefix}_수급모멘텀"] = net.rolling(5).sum() - net.rolling(10).sum()

        # Supply disparity = institutional + foreign - retail
        g["수급괴리도"] = (g["기관계_순매수_5MA"].fillna(0) +
                       g["외국인_순매수_5MA"].fillna(0) -
                       g["개인_순매수_5MA"].fillna(0))

        # Retail buying ratio
        total = (g["개인_매수2_거래대금"].fillna(0) +
                 g["외국인_매수2_거래대금"].fillna(0) +
                 g["기관계_매수2_거래대금"].fillna(0)).replace(0, np.nan)
        g["개인매수비중"] = (g["개인_매수2_거래대금"].fillna(0) / total * 100).round(1)

        results.append(g)

    df = pd.concat(results, ignore_index=True)
    print(f"  Supply/demand features added: net buying, 5/10MA, momentum, disparity")
    return df


def _pick_col(df, *candidates):
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column. Tried: {candidates}")


def load_index_data(path):
    """
    Load data/index.csv and build daily KOSPI columns only.
    Expected columns (current crawler output):
      지수코드, 지수명, 날짜, 종가, 시가, 고가, 저가
    """
    idx = pd.read_csv(path)
    if len(idx) == 0:
        return pd.DataFrame(columns=["date"])

    code_col = _pick_col(idx, "지수코드")
    date_col = _pick_col(idx, "날짜")
    close_col = _pick_col(idx, "종가")
    high_col = _pick_col(idx, "고가")
    low_col = _pick_col(idx, "저가")

    idx = idx.copy()
    idx[code_col] = idx[code_col].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(4)
    idx[date_col] = pd.to_datetime(idx[date_col].astype(str), format="mixed", errors="coerce")
    idx = idx[idx[date_col].notna()].copy()

    # Keep only KOSPI(0001)
    idx = idx[idx[code_col] == "0001"].copy()
    if len(idx) == 0:
        return pd.DataFrame(columns=["date"])

    base = idx[[date_col]].drop_duplicates().rename(columns={date_col: "date"}).sort_values("date")

    def _merge_one(code: str, prefix: str, frame: pd.DataFrame) -> pd.DataFrame:
        sub = idx[idx[code_col] == code][[date_col, close_col, high_col, low_col]].copy()
        sub = sub.rename(
            columns={
                date_col: "date",
                close_col: f"{prefix}_close",
                high_col: f"{prefix}_high",
                low_col: f"{prefix}_low",
            }
        )
        return frame.merge(sub, on="date", how="left")

    out = _merge_one("0001", "index_kospi", base)
    print(f"  Index data: {len(out):,} days (KOSPI)")
    return out


def merge_index_features(features_df, index_df):
    """Attach daily index columns to stock feature rows by date."""
    if index_df is None or len(index_df) == 0:
        print("  Index merge skipped: empty index dataframe")
        return features_df

    out = features_df.copy()
    date_col = _pick_col(out, "날짜")
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    merged = out.merge(index_df, on="date", how="left")
    print(
        f"  Index merged: {len(merged):,} rows "
        f"(KOSPI present={(merged['index_kospi_close'].notna().sum() if 'index_kospi_close' in merged.columns else 0):,})"
    )
    return merged


def _gap_ratio(buy: pd.Series, sell: pd.Series) -> np.ndarray:
    denom = (buy + sell) / 2
    return np.where(denom == 0, np.nan, np.abs(buy - sell) / denom)


def add_approx_totals_and_normalized_flows(df, gap_thresh: float = 0.05):
    """
    Approximate total volume/amount from 3 investor groups and add normalized flow features.
    No official total columns required.
    """
    out = df.copy()

    p_bv = _pick_col(out, "개인_매수2_거래량")
    p_ba = _pick_col(out, "개인_매수2_거래대금")
    p_sv = _pick_col(out, "개인_매도_거래량")
    p_sa = _pick_col(out, "개인_매도_거래대금")
    f_bv = _pick_col(out, "외국인_매수2_거래량")
    f_ba = _pick_col(out, "외국인_매수2_거래대금")
    f_sv = _pick_col(out, "외국인_매도_거래량")
    f_sa = _pick_col(out, "외국인_매도_거래대금")
    i_bv = _pick_col(out, "기관계_매수2_거래량")
    i_ba = _pick_col(out, "기관계_매수2_거래대금")
    i_sv = _pick_col(out, "기관계_매도_거래량")
    i_sa = _pick_col(out, "기관계_매도_거래대금")

    out["approx_total_vol_buy"] = out[p_bv].fillna(0) + out[f_bv].fillna(0) + out[i_bv].fillna(0)
    out["approx_total_vol_sell"] = out[p_sv].fillna(0) + out[f_sv].fillna(0) + out[i_sv].fillna(0)
    out["approx_total_vol"] = (out["approx_total_vol_buy"] + out["approx_total_vol_sell"]) / 2

    out["approx_total_amt_buy"] = out[p_ba].fillna(0) + out[f_ba].fillna(0) + out[i_ba].fillna(0)
    out["approx_total_amt_sell"] = out[p_sa].fillna(0) + out[f_sa].fillna(0) + out[i_sa].fillna(0)
    out["approx_total_amt"] = (out["approx_total_amt_buy"] + out["approx_total_amt_sell"]) / 2

    out["vol_gap_ratio"] = _gap_ratio(out["approx_total_vol_buy"], out["approx_total_vol_sell"])
    out["amt_gap_ratio"] = _gap_ratio(out["approx_total_amt_buy"], out["approx_total_amt_sell"])

    out["inst_net_amt_calc"] = out[i_ba].fillna(0) - out[i_sa].fillna(0)
    out["foreign_net_amt_calc"] = out[f_ba].fillna(0) - out[f_sa].fillna(0)
    out["retail_net_amt_calc"] = out[p_ba].fillna(0) - out[p_sa].fillna(0)

    denom_amt = out["approx_total_amt"].replace(0, np.nan)
    out["inst_flow_ratio_calc"] = out["inst_net_amt_calc"] / denom_amt
    out["foreign_flow_ratio_calc"] = out["foreign_net_amt_calc"] / denom_amt
    out["smart_flow_ratio_calc"] = (out["inst_net_amt_calc"] + out["foreign_net_amt_calc"]) / denom_amt

    bad_gap = (out["vol_gap_ratio"] > gap_thresh) | (out["amt_gap_ratio"] > gap_thresh)
    cols_to_null = [
        "approx_total_vol_buy", "approx_total_vol_sell", "approx_total_vol",
        "approx_total_amt_buy", "approx_total_amt_sell", "approx_total_amt",
        "inst_flow_ratio_calc", "foreign_flow_ratio_calc", "smart_flow_ratio_calc",
    ]
    out.loc[bad_gap, cols_to_null] = np.nan
    print("  Added approx totals + normalized flow features")
    return out


def add_market_regime_features(df):
    """
    Add market regime / relative strength features using merged index.csv.
    Benchmark is fixed to KOSPI (main index) for all stocks.
    """
    out = df.copy()
    stock_col = _pick_col(out, "종목코드")
    date_col = _pick_col(out, "날짜")
    close_col = _pick_col(out, "종가")
    high_col = _pick_col(out, "최고가")

    needed = {"index_kospi_close", "index_kospi_high", "index_kospi_low"}
    if not needed.issubset(set(out.columns)):
        print("  Market regime features skipped: no index columns")
        return out

    out = out.sort_values([stock_col, date_col]).copy()
    out[stock_col] = out[stock_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    # Use KOSPI as the single benchmark index for all stocks.
    out["index_close"] = out["index_kospi_close"]
    out["index_high"] = out["index_kospi_high"]
    out["index_low"] = out["index_kospi_low"]

    out["ret_1"] = out.groupby(stock_col)[close_col].pct_change(1)
    out["ret_5"] = out.groupby(stock_col)[close_col].pct_change(5)
    out["ret_20"] = out.groupby(stock_col)[close_col].pct_change(20)

    idx = (
        out[[date_col, "index_close", "index_high", "index_low"]]
        .drop_duplicates(date_col)
        .sort_values(date_col)
        .copy()
    )
    idx["index_ret_1"] = idx["index_close"].pct_change(1)
    idx["index_ret_5"] = idx["index_close"].pct_change(5)
    idx["index_ret_20"] = idx["index_close"].pct_change(20)
    idx["index_intraday_vol"] = (idx["index_high"] - idx["index_low"]) / idx["index_close"].replace(0, np.nan)
    idx["index_vol_20"] = idx["index_ret_1"].rolling(20).std()

    out = out.merge(
        idx[[date_col, "index_ret_1", "index_ret_5", "index_ret_20", "index_intraday_vol", "index_vol_20"]],
        on=date_col,
        how="left",
    )

    out["relative_strength_5"] = out["ret_5"] - out["index_ret_5"]
    out["relative_strength_20"] = out["ret_20"] - out["index_ret_20"]
    out["volatility_20"] = out.groupby(stock_col)["ret_1"].transform(lambda s: s.rolling(20).std())

    if "approx_total_amt" in out.columns:
        out["amt_ma60"] = out.groupby(stock_col)["approx_total_amt"].transform(lambda s: s.rolling(60).mean())
        out["amt_std60"] = out.groupby(stock_col)["approx_total_amt"].transform(lambda s: s.rolling(60).std())
        out["volume_z"] = (out["approx_total_amt"] - out["amt_ma60"]) / out["amt_std60"].replace(0, np.nan)

    if "MA_60" in out.columns:
        out["ma60_slope"] = out.groupby(stock_col)["MA_60"].diff(5)
        out["above_ma60"] = (out[close_col] > out["MA_60"]).astype(int)

    out["rebound_3d"] = (out[close_col] > out.groupby(stock_col)[close_col].shift(3)).astype(int)
    out["high_break"] = (out[close_col] > out.groupby(stock_col)[high_col].shift(1)).astype(int)
    roll_max_20 = out.groupby(stock_col)[close_col].transform(lambda s: s.rolling(20).max())
    out["drawdown_20"] = out[close_col] / roll_max_20.replace(0, np.nan) - 1

    print("  Added market regime / relative strength features")
    return out


# ═══════════════════════════════════════════════════════════
# STEP 5: News Cleaning
# ═══════════════════════════════════════════════════════════

# Patterns to remove
MARKET_PATTERNS = [
    r"거래량\s*상위", r"외국인\s*순매수", r"외국인\s*순매도",
    r"기관\s*순매수", r"기관\s*순매도", r"수익률\s*상위",
    r"상한가", r"하한가", r"52주\s*(신)?고가",
]
TABLE_PATTERNS = [r"^\[표\]", r"^\[게시판\]", r"부고\s*[-·]", r"인사\s*[-·]"]
AD_PATTERNS = [r"광고", r"\bPR\b", r"스폰서", r"제공\s*[:：]"]
SHORT_TITLE_LEN = 8

# Notebook pre-filter (market status/meta headlines)
NOTEBOOK_TITLE_FILTER_PATTERN = (
    r"^\[정오\s*시황\]"
    r"|"
    r"^(?:오전|오후)\s*\d{1,2}:\d{2}\s*(?:현재\s*)?"
    r"(?:코스피|코스닥).*(?:매수우위|매도우위|매수강세|매도강세)"
    r"|"
    r"^\[(?:포토|인사|오늘의\s*IR|한경유레카)\]"
    r"|"
    r"<\s*(?:유|코)\s*>"
    r"|"
    r"^오늘의\s*메모"
)


def prepare_news_input(df):
    """Normalize merged all_news.csv or raw news CSV into preprocess input schema."""
    out = df.copy()

    # Normalize stock code column candidates
    if "stock_code" not in out.columns:
        if "primary_stock_code" in out.columns:
            out["stock_code"] = out["primary_stock_code"]
        else:
            raise KeyError("news input requires 'stock_code' or 'primary_stock_code'")

    # Normalize date column candidates (all_news.csv commonly uses published_dt)
    if "date" not in out.columns:
        if "published_dt" in out.columns:
            out["date"] = out["published_dt"]
        elif "published_at" in out.columns:
            out["date"] = out["published_at"].astype(str).str.slice(0, 8)
        else:
            raise KeyError("news input requires 'date' or 'published_dt'/'published_at'")

    if "title" not in out.columns:
        raise KeyError("news input requires 'title' column")

    out["stock_code"] = (
        out["stock_code"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(6)
    )
    out["date"] = pd.to_datetime(out["date"].astype(str).str.slice(0, 10), errors="coerce")
    out["title"] = out["title"].fillna("").astype(str)

    # Drop rows that cannot be merged later
    out = out[(out["stock_code"].str.len() == 6) & out["date"].notna()].copy()
    return out


def notebook_style_prefilter_news(df):
    """Remove market-status/meta headlines previously filtered in basic.ipynb."""
    if "title" not in df.columns or len(df) == 0:
        return df
    n_before = len(df)
    mask = ~df["title"].str.contains(NOTEBOOK_TITLE_FILTER_PATTERN, regex=True, na=False)
    out = df[mask].copy()
    print(f"  Notebook-style title filter: {n_before:,} -> {len(out):,} ({n_before-len(out):,} removed)")
    return out


def clean_news(df):
    """Clean news data: remove market noise, tables, ads, duplicates, control chars."""
    n_before = len(df)

    # Remove control characters
    df["title"] = df["title"].fillna("").astype(str).apply(
        lambda x: re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", x).strip()
    )

    # Remove market summary articles
    market_mask = df["title"].str.contains("|".join(MARKET_PATTERNS), na=False, regex=True)

    # Remove table/bulletin/obituary
    table_mask = df["title"].str.contains("|".join(TABLE_PATTERNS), na=False, regex=True)

    # Remove ads
    ad_mask = df["title"].str.contains("|".join(AD_PATTERNS), na=False, regex=True)

    # Remove too-short titles
    short_mask = df["title"].str.len() < SHORT_TITLE_LEN

    # Remove nulls
    null_mask = df["title"].isna() | (df["title"].str.strip() == "")

    # Apply all filters
    remove = market_mask | table_mask | ad_mask | short_mask | null_mask
    df = df[~remove].copy()

    # Remove duplicates (same stock + same date + same title)
    df = df.drop_duplicates(subset=["stock_code", "date", "title"])

    n_after = len(df)
    print(f"  News cleaning: {n_before:,} → {n_after:,} ({n_before - n_after:,} removed)")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 6: Korean Financial Sentiment Analysis
# ═══════════════════════════════════════════════════════════

# Positive keywords (weight)
POS_KEYWORDS = {
    # Earnings
    "호실적": 1.5, "어닝서프라이즈": 1.5, "실적개선": 1.2, "매출증가": 1.0,
    "영업이익": 0.8, "순이익": 0.8, "흑자전환": 1.5, "흑자": 1.0,
    "사상최대": 1.5, "역대최대": 1.5, "최대실적": 1.5,
    # Price
    "급등": 1.0, "강세": 0.8, "상승": 0.5, "반등": 0.8,
    "신고가": 1.0, "돌파": 0.7, "랠리": 0.8,
    # Analyst
    "목표가상향": 1.5, "매수": 0.8, "비중확대": 1.0, "투자의견상향": 1.2,
    # Business
    "수주": 1.2, "계약": 0.8, "신사업": 1.0, "혁신": 0.8,
    "인수": 0.7, "합병": 0.5, "제휴": 0.7, "협력": 0.5,
    "증설": 1.0, "투자확대": 0.8, "생산능력": 0.7,
    # Dividends
    "배당": 0.8, "배당확대": 1.2, "자사주": 1.0, "주주환원": 1.0,
    # Upgrade
    "성장": 0.7, "확대": 0.5, "개선": 0.7, "회복": 0.8,
}

# Negative keywords (weight)
NEG_KEYWORDS = {
    # Earnings
    "실적악화": 1.5, "적자": 1.2, "적자전환": 1.5, "어닝쇼크": 1.5,
    "매출감소": 1.0, "이익감소": 1.0, "실적부진": 1.2,
    # Price
    "급락": 1.0, "폭락": 1.2, "하락": 0.5, "약세": 0.5,
    "신저가": 1.0, "52주최저": 1.0,
    # Risk
    "리콜": 1.2, "소송": 1.0, "벌금": 1.0, "제재": 1.0,
    "규제": 0.8, "조사": 0.7, "압수수색": 1.5,
    # Analyst
    "목표가하향": 1.5, "매도": 0.8, "비중축소": 1.0, "투자의견하향": 1.2,
    # Business
    "부채": 0.8, "차입": 0.5, "유상증자": 1.0, "감자": 1.2,
    "구조조정": 1.0, "감원": 0.8, "폐쇄": 1.0,
    # Downgrade
    "둔화": 0.7, "축소": 0.5, "악화": 0.8, "위기": 0.8,
}


def score_sentiment(title):
    """Score a single news title using keyword matching + numeric patterns."""
    if not isinstance(title, str) or len(title.strip()) == 0:
        return 0.0

    pos_score = 0.0
    neg_score = 0.0

    # Keyword matching
    for kw, w in POS_KEYWORDS.items():
        if kw in title:
            pos_score += w

    for kw, w in NEG_KEYWORDS.items():
        if kw in title:
            neg_score += w

    # Numeric patterns: "XX% 증가/상승" → positive
    inc_match = re.findall(r"(\d+\.?\d*)\s*%?\s*(증가|상승|↑|개선|확대|성장)", title)
    for m in inc_match:
        pct = float(m[0])
        if pct >= 100:
            pos_score += 1.5
        elif pct >= 50:
            pos_score += 1.0
        elif pct >= 10:
            pos_score += 0.5

    dec_match = re.findall(r"(\d+\.?\d*)\s*%?\s*(감소|하락|↓|악화|축소|둔화)", title)
    for m in dec_match:
        pct = float(m[0])
        if pct >= 100:
            neg_score += 1.5
        elif pct >= 50:
            neg_score += 1.0
        elif pct >= 10:
            neg_score += 0.5

    # Target price patterns
    if re.search(r"목표가\s*(↑|상향|인상)", title):
        pos_score += 1.5
    if re.search(r"목표가\s*(↓|하향|인하)", title):
        neg_score += 1.5

    # Normalize to [-1, 1]
    total = pos_score + neg_score
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos_score - neg_score) / max(total, 1.0)))


def analyze_sentiment(news_df):
    """Add sentiment scores to news dataframe."""
    news_df["sentiment"] = news_df["title"].apply(score_sentiment)
    pos = (news_df["sentiment"] > 0).sum()
    neg = (news_df["sentiment"] < 0).sum()
    neu = (news_df["sentiment"] == 0).sum()
    print(f"  Sentiment: {pos:,} positive ({pos/len(news_df)*100:.1f}%), "
          f"{neg:,} negative ({neg/len(news_df)*100:.1f}%), "
          f"{neu:,} neutral ({neu/len(news_df)*100:.1f}%)")
    return news_df


# ═══════════════════════════════════════════════════════════
# STEP 7: Aggregate Daily Sentiment
# ═══════════════════════════════════════════════════════════

def aggregate_daily_sentiment(news_df):
    """Aggregate news sentiment per stock per day."""
    news_df["date"] = pd.to_datetime(news_df["date"])

    daily = news_df.groupby(["stock_code", "date"]).agg(
        news_count=("sentiment", "count"),
        sent_mean=("sentiment", "mean"),
        sent_pos_ratio=("sentiment", lambda x: (x > 0).mean()),
        sent_neg_ratio=("sentiment", lambda x: (x < 0).mean()),
        sent_max=("sentiment", "max"),
        sent_min=("sentiment", "min"),
    ).reset_index()

    # 3-day moving average and momentum
    dfs = []
    for code, g in daily.groupby("stock_code"):
        g = g.sort_values("date").copy()
        g["sent_3ma"] = g["sent_mean"].rolling(3, min_periods=1).mean()
        g["sent_momentum"] = g["sent_mean"] - g["sent_mean"].shift(5).fillna(0)
        dfs.append(g)

    daily = pd.concat(dfs, ignore_index=True)
    print(f"  Daily aggregation: {len(daily):,} stock-days with news")
    return daily


# ═══════════════════════════════════════════════════════════
# STEP 8: Merge Everything
# ═══════════════════════════════════════════════════════════

def merge_features_and_news(features_df, daily_sent):
    """Merge price features with daily sentiment. Fill missing days with 0."""
    # Prepare merge keys
    features_df["stock_code"] = (
        features_df["종목코드"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    )
    features_df["date"] = features_df["날짜"].dt.strftime("%Y-%m-%d")
    daily_sent["date"] = daily_sent["date"].dt.strftime("%Y-%m-%d")
    daily_sent["stock_code"] = (
        daily_sent["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    )

    merged = features_df.merge(
        daily_sent, on=["stock_code", "date"], how="left"
    )

    # Fill missing news days with 0
    news_cols = ["news_count", "sent_mean", "sent_pos_ratio", "sent_neg_ratio",
                 "sent_max", "sent_min", "sent_3ma", "sent_momentum"]
    for col in news_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print(f"  Merged: {len(merged):,} rows "
          f"({(merged['news_count'] > 0).sum():,} with news, "
          f"{(merged['news_count'] == 0).sum():,} without)")
    return merged


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data → features CSV")
    parser.add_argument("--price", type=str, default=DEFAULT_PRICE_INPUT,
                        help="Raw price/supply CSV path")
    parser.add_argument("--news", type=str, default=DEFAULT_NEWS_INPUT,
                        help="Raw news CSV path (set 'none' to skip)")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX_INPUT,
                        help="Index CSV path (data/index.csv from index_chart)")
    parser.add_argument("--output", type=str, default=DEFAULT_FEATURES_OUTPUT,
                        help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)
    if "__TODO_" in args.price or ("__TODO_" in args.news and args.news.lower() != "none"):
        print("  NOTE: Placeholder input paths are set. Please update --price / --news.")

    # Step 1-2: Load price data
    print("\n[1/8] Loading price data...")
    df = load_price_data(args.price)

    # Step 3: Technical indicators
    print("\n[2/8] Computing technical indicators...")
    df = add_technical_indicators(df)

    # Step 4: Supply/demand features
    print("\n[3/8] Computing supply/demand features...")
    df = add_supply_features(df)

    # Merge index data + add additional features from build_features_v2 guide
    try:
        print(f"\n[3.5/8] Loading index data ({args.index})...")
        index_df = load_index_data(args.index)
        if len(index_df) > 0:
            df = merge_index_features(df, index_df)
            print("\n[3.6/8] Computing approx totals / normalized flows...")
            df = add_approx_totals_and_normalized_flows(df)
            print("\n[3.7/8] Computing market regime features...")
            df = add_market_regime_features(df)
    except FileNotFoundError:
        print(f"  WARNING: {args.index} not found. Skipping index-based features.")

    # Step 5-7: News (optional)
    if args.news.lower() != "none":
        print(f"\n[4/8] Loading news data ({args.news})...")
        try:
            news_raw = pd.read_csv(args.news)
            print(f"  Raw news loaded: {len(news_raw):,} rows")
            news_raw = prepare_news_input(news_raw)
            print(f"  Normalized news input: {len(news_raw):,} rows")

            # Integrate the basic.ipynb simple title filter into preprocess pipeline
            news_raw = notebook_style_prefilter_news(news_raw)

            print("\n[5/8] Cleaning news...")
            news_clean = clean_news(news_raw)

            print("\n[6/8] Sentiment analysis...")
            news_sent = analyze_sentiment(news_clean)

            print("\n[7/8] Aggregating daily sentiment...")
            daily_sent = aggregate_daily_sentiment(news_sent)

            print("\n[8/8] Merging features + news...")
            df = merge_features_and_news(df, daily_sent)
        except FileNotFoundError:
            print(f"  WARNING: {args.news} not found. Skipping news features.")
    else:
        print("\n[4-7] Skipping news (--news none)")

    # Save
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 60}")
    print(f"  DONE: {args.output} ({len(df):,} rows, {len(df.columns)} columns)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
