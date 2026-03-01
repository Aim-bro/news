"""
=============================================================
probability.py ??Inflection Point Probability Calculator
=============================================================
Core engine: calculates P(high) and P(low) for date t-3
given data up to date t. No future information used.
=============================================================
"""

import numpy as np
import pandas as pd
try:
    from .config import *
except ImportError:
    from config import *


def safe(v, default=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def safe_mean(series, default=0):
    if len(series) == 0:
        return default
    m = series.mean()
    return default if np.isnan(m) else m


def calc_probability(g, target_idx):
    """
    Calculate inflection point probability for target_idx (= today - DELAY).

    Args:
        g: DataFrame for one stock, sorted by date, reset_index.
        target_idx: index of the candidate inflection date.

    Returns:
        dict with 'high_prob', 'low_prob', metadata. Or None if insufficient data.
    """
    close = g["종가"].values
    n = len(g)
    today_idx = target_idx + DELAY

    if today_idx >= n or target_idx < LOOKBACK:
        return None

    tp = close[target_idx]
    todayp = close[today_idx]
    left = close[max(0, target_idx - LOOKBACK) : target_idx]
    after = close[target_idx + 1 : today_idx + 1]

    if len(left) == 0 or len(after) == 0:
        return None

    row = g.iloc[target_idx]
    # RSI_14: 14일 RSI (과매수/과매도 판단)
    # 이격도_20: 종가와 20일 이동평균의 괴리율(%)
    # BB_PctB: 볼린저 밴드 내 위치 (0=하단, 1=상단, 1초과=상단 돌파)
    rsi = safe(row.get("RSI_14", 50), 50)
    gap20 = safe(row.get("이격도_20", 100), 100)
    bb = safe(row.get("BB_PctB", 0.5), 0.5)

    # Supply/demand flip
    # 개인/외국인/기관계_순매수(거래대금 기준)의 전후 평균 변화량으로 수급 전환 강도 측정
    sl = max(0, target_idx - 5)
    pre_for = safe_mean(g.iloc[sl:target_idx]["외국인_순매수"])
    post_for = safe_mean(g.iloc[target_idx + 1 : today_idx + 1]["외국인_순매수"])
    pre_inst = safe_mean(g.iloc[sl:target_idx]["기관계_순매수"])
    post_inst = safe_mean(g.iloc[target_idx + 1 : today_idx + 1]["기관계_순매수"])
    pre_ret = safe_mean(g.iloc[sl:target_idx]["개인_순매수"])
    post_ret = safe_mean(g.iloc[target_idx + 1 : today_idx + 1]["개인_순매수"])

    ff = post_for - pre_for
    iff = post_inst - pre_inst
    rf = post_ret - pre_ret
    decline = (tp - todayp) / tp * 100
    rise = (todayp - tp) / tp * 100

    # News sentiment (뉴스 보너스/패널티 계산용)
    # news_count: 기사 수(뉴스 존재 여부)
    # sent_mean: 평균 감성 점수(-1~+1)
    # sent_momentum: 감성 변화량(최근 방향 전환 확인)
    nw = g.iloc[max(0, target_idx - 2) : today_idx + 1]
    nc_col = "news_count" if "news_count" in g.columns else None
    sm_col = "sent_mean" if "sent_mean" in g.columns else None
    smom_col = "sent_momentum" if "sent_momentum" in g.columns else None

    ncnt = int(nw[nc_col].sum()) if nc_col else 0
    # sent_mean의 0은 뉴스 없음/placeholder인 경우가 있어 평균 계산에서 제외
    sent = safe(nw[sm_col].replace(0, np.nan).mean(), 0) if sm_col else 0
    smom = safe(row.get("sent_momentum", 0), 0) if smom_col else 0
    has_news = ncnt > 0
    # Market regime features (preprocess.py generated; may be NaN)
    # relative_strength_5: 종목 5일 수익률 - 코스피 5일 수익률 (상대강도)
    # index_ret_5: 코스피 5일 수익률 (시장 위험선호/회피 분위기)
    # volume_z: 근사 거래대금의 60일 z-score (이벤트 강도/과열/투매 흔적)
    rs5 = safe(row.get("relative_strength_5", np.nan), np.nan)
    idx_ret5 = safe(row.get("index_ret_5", np.nan), np.nan)
    volz = safe(row.get("volume_z", np.nan), np.nan)
    # ------------------------------
    # HIGH (peak) probability
    # ------------------------------
    hs = 0.0
    if tp >= left.max():        hs += W_LEFT_EXTREME
    if tp >= after.max():       hs += W_RIGHT_EXTREME
    if decline >= 5:            hs += W_MOVE_STRONG
    elif decline >= 2:          hs += W_MOVE_MILD
    if rsi >= 75:               hs += W_RSI_EXTREME
    elif rsi >= 70:             hs += W_RSI_STRONG
    elif rsi >= 65:             hs += W_RSI_MILD
    if gap20 >= 115:            hs += W_GAP_EXTREME
    elif gap20 >= 110:          hs += W_GAP_STRONG
    elif gap20 >= 105:          hs += W_GAP_MILD
    if bb >= 1.05:              hs += W_BB_EXTREME
    elif bb >= 0.95:            hs += W_BB_STRONG
    if iff < -30:               hs += W_INST_STRONG
    elif iff < -10:             hs += W_INST_MID
    elif iff < -3:              hs += W_INST_MILD
    if ff < -30:                hs += W_FOR_STRONG
    elif ff < -10:              hs += W_FOR_MID
    elif ff < -3:               hs += W_FOR_MILD
    if rf > 30:                 hs += W_RET_STRONG
    elif rf > 10:               hs += W_RET_MID
    if rsi >= 70 and iff < -10: hs += W_COMBO_RSI_INST
    if tp >= left.max() and decline >= 3 and iff < 0:
                                hs += W_COMBO_PRICE_INST

    # News bonus/penalty for HIGH
    nb_h = 0.0
    if has_news:
        if sent <= -0.3:    nb_h += NEWS_BONUS_STRONG
        elif sent <= -0.1:  nb_h += NEWS_BONUS_MILD
        if smom <= -0.3:    nb_h += NEWS_BONUS_MOM_STRONG
        elif smom <= -0.1:  nb_h += NEWS_BONUS_MOM_MILD
        if sent >= 0.3:     nb_h -= NEWS_PENALTY_STRONG
        elif sent >= 0.1:   nb_h -= NEWS_PENALTY_MILD
    # ------------------------------
    # LOW (bottom) probability
    # ------------------------------
    ls = 0.0
    if tp <= left.min():        ls += W_LEFT_EXTREME
    if tp <= after.min():       ls += W_RIGHT_EXTREME
    if rise >= 5:               ls += W_MOVE_STRONG
    elif rise >= 2:             ls += W_MOVE_MILD
    if rsi <= 25:               ls += W_RSI_EXTREME
    elif rsi <= 30:             ls += W_RSI_STRONG
    elif rsi <= 35:             ls += W_RSI_MILD
    if gap20 <= 85:             ls += W_GAP_EXTREME
    elif gap20 <= 90:           ls += W_GAP_STRONG
    elif gap20 <= 95:           ls += W_GAP_MILD
    if bb <= -0.05:             ls += W_BB_EXTREME
    elif bb <= 0.05:            ls += W_BB_STRONG
    if iff > 30:                ls += W_INST_STRONG
    elif iff > 10:              ls += W_INST_MID
    elif iff > 3:               ls += W_INST_MILD
    if ff > 30:                 ls += W_FOR_STRONG
    elif ff > 10:               ls += W_FOR_MID
    elif ff > 3:                ls += W_FOR_MILD
    if rf < -30:                ls += W_RET_STRONG
    elif rf < -10:              ls += W_RET_MID
    if rsi <= 30 and iff > 10:  ls += W_COMBO_RSI_INST
    if tp <= left.min() and rise >= 3 and iff > 0:
                                ls += W_COMBO_PRICE_INST

    # News bonus/penalty for LOW
    nb_l = 0.0
    if has_news:
        if sent >= 0.3:     nb_l += NEWS_BONUS_STRONG
        elif sent >= 0.1:   nb_l += NEWS_BONUS_MILD
        if smom >= 0.3:     nb_l += NEWS_BONUS_MOM_STRONG
        elif smom >= 0.1:   nb_l += NEWS_BONUS_MOM_MILD
        if sent <= -0.3:    nb_l -= NEWS_PENALTY_STRONG
        elif sent <= -0.1:  nb_l -= NEWS_PENALTY_MILD

    # Market regime bonus/penalty (small overlay on top of base+news)
    rb_h = 0.0
    rb_l = 0.0

    if not pd.isna(rs5):
        # Relative strength up => peak(고점) 쪽 가점 / bottom(저점) 쪽 감점
        if rs5 >= 0.10:
            rb_h += REGIME_RS_BONUS_STRONG
            rb_l -= REGIME_RS_PENALTY_STRONG
        elif rs5 >= 0.03:
            rb_h += REGIME_RS_BONUS_MILD
            rb_l -= REGIME_RS_PENALTY_MILD
        elif rs5 <= -0.10:
            rb_l += REGIME_RS_BONUS_STRONG
            rb_h -= REGIME_RS_PENALTY_STRONG
        elif rs5 <= -0.03:
            rb_l += REGIME_RS_BONUS_MILD
            rb_h -= REGIME_RS_PENALTY_MILD

    if not pd.isna(idx_ret5):
        # Risk-on market tends to extend highs; risk-off tends to reinforce bottoms/capitulation
        if idx_ret5 >= 0.05:
            rb_h += REGIME_INDEX_BONUS_STRONG
            rb_l -= REGIME_INDEX_PENALTY_STRONG
        elif idx_ret5 >= 0.02:
            rb_h += REGIME_INDEX_BONUS_MILD
            rb_l -= REGIME_INDEX_PENALTY_MILD
        elif idx_ret5 <= -0.05:
            rb_l += REGIME_INDEX_BONUS_STRONG
            rb_h -= REGIME_INDEX_PENALTY_STRONG
        elif idx_ret5 <= -0.02:
            rb_l += REGIME_INDEX_BONUS_MILD
            rb_h -= REGIME_INDEX_PENALTY_MILD

    if not pd.isna(volz):
        # High volume confirms inflection significance (either blow-off top or capitulation bottom)
        if volz >= 2.0:
            rb_h += REGIME_VOLZ_BONUS_STRONG
            rb_l += REGIME_VOLZ_BONUS_STRONG
        elif volz >= 1.0:
            rb_h += REGIME_VOLZ_BONUS_MILD
            rb_l += REGIME_VOLZ_BONUS_MILD
        elif volz <= -1.0:
            rb_h -= REGIME_VOLZ_PENALTY_LOW_LIQ
            rb_l -= REGIME_VOLZ_PENALTY_LOW_LIQ

    final_h = max(min(hs + nb_h + rb_h, 1.0), 0)
    final_l = max(min(ls + nb_l + rb_l, 1.0), 0)

    return {
        "high_prob": final_h,
        "low_prob": final_l,
        "base_high": hs,
        "base_low": ls,
        "news_bonus_h": round(nb_h, 3),
        "news_bonus_l": round(nb_l, 3),
        "regime_bonus_h": round(rb_h, 3),
        "regime_bonus_l": round(rb_l, 3),
        "target_price": tp,
        "today_price": todayp,
        "rsi": round(rsi, 1),
        "gap20": round(gap20, 1),
        "bb": round(bb, 3),
        "decline_pct": round(decline, 2),
        "rise_pct": round(rise, 2),
        "inst_flip": round(iff, 1),
        "for_flip": round(ff, 1),
        "ret_flip": round(rf, 1),
        "relative_strength_5": None if pd.isna(rs5) else round(rs5, 4),
        "index_ret_5": None if pd.isna(idx_ret5) else round(idx_ret5, 4),
        "volume_z": None if pd.isna(volz) else round(volz, 3),
        "sentiment": round(sent, 3),
        "sent_momentum": round(smom, 3),
        "news_count": ncnt,
    }




