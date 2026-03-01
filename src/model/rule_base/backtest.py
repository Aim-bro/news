"""
=============================================================
backtest.py — Backtest Engine
=============================================================
Simulates buy/sell trades based on probability signals.
Supports trailing stop loss.
=============================================================
"""

import pandas as pd
import numpy as np
try:
    from .config import *
    from .probability import calc_probability
except ImportError:
    from config import *
    from probability import calc_probability


def run_backtest(features_df, threshold_buy=THRESHOLD_BUY, threshold_sell=THRESHOLD_SELL,
                 trailing_stop=TRAILING_STOP_PCT):
    """
    Run backtest across all stocks.

    Returns:
        trades_df: DataFrame of all trades
        signals: list of all daily signals
    """
    all_trades = []
    all_signals = []

    for code, group in features_df.groupby("종목코드"):
        name = group["종목명"].iloc[0]
        g = group.sort_values("날짜").reset_index(drop=True)
        close = g["종가"].values
        dates = g["날짜"].values
        n = len(g)

        position = None
        entry_price = 0
        entry_date = None
        entry_info = {}
        peak_price = 0  # for trailing stop

        for today in range(LOOKBACK + DELAY + 2, n):
            target = today - DELAY
            result = calc_probability(g, target)
            if result is None:
                continue

            hp = result["high_prob"]
            lp = result["low_prob"]

            sig = {
                "stock": name,
                "today": str(dates[today])[:10],
                "target_date": str(dates[target])[:10],
                "high_prob": round(hp * 100, 1),
                "low_prob": round(lp * 100, 1),
                "rsi": result["rsi"],
                "sentiment": result["sentiment"],
            }
            all_signals.append(sig)

            # ── Trailing stop check ──
            if position == "long" and trailing_stop > 0:
                current_price = close[today]
                peak_price = max(peak_price, current_price)
                drawdown = (peak_price - current_price) / peak_price
                if drawdown >= trailing_stop:
                    if today + 1 < n:
                        exit_price = close[today + 1]
                        exit_date = dates[today + 1]
                        ret = (exit_price - entry_price) / entry_price * 100
                        hold = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
                        all_trades.append(_make_trade(
                            name, entry_date, exit_date, entry_price, exit_price,
                            ret, hold, entry_info, 0, "TRAILING_STOP"
                        ))
                        position = None
                        continue

            # ── BUY signal ──
            if lp >= threshold_buy and position is None:
                if today + 1 < n:
                    entry_price = close[today + 1]
                    entry_date = dates[today + 1]
                    peak_price = entry_price
                    position = "long"
                    entry_info = {
                        "prob": round(lp * 100, 1),
                        "base": round(result["base_low"] * 100, 1),
                        "bonus": round(result["news_bonus_l"] * 100, 1),
                        "rsi": result["rsi"],
                        "sent": result["sentiment"],
                        "ncnt": result["news_count"],
                    }

            # ── SELL signal ──
            elif hp >= threshold_sell and position == "long":
                if today + 1 < n:
                    exit_price = close[today + 1]
                    exit_date = dates[today + 1]
                    ret = (exit_price - entry_price) / entry_price * 100
                    hold = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
                    all_trades.append(_make_trade(
                        name, entry_date, exit_date, entry_price, exit_price,
                        ret, hold, entry_info, round(hp * 100, 1), "SIGNAL"
                    ))
                    position = None

        # Unclosed position
        if position == "long":
            exit_price = close[-1]
            ret = (exit_price - entry_price) / entry_price * 100
            hold = (pd.Timestamp(dates[-1]) - pd.Timestamp(entry_date)).days
            all_trades.append(_make_trade(
                name, entry_date, dates[-1], entry_price, exit_price,
                ret, hold, entry_info, 0, "OPEN"
            ))

    return pd.DataFrame(all_trades), all_signals


def _make_trade(name, entry_date, exit_date, entry_price, exit_price,
                ret, hold, entry_info, sell_prob, exit_type):
    return {
        "stock": name,
        "entry_date": str(entry_date)[:10],
        "exit_date": str(exit_date)[:10],
        "entry_price": int(entry_price),
        "exit_price": int(exit_price),
        "return_pct": round(ret, 2),
        "hold_days": hold,
        "buy_prob": entry_info.get("prob", 0),
        "buy_base": entry_info.get("base", 0),
        "buy_bonus": entry_info.get("bonus", 0),
        "buy_rsi": entry_info.get("rsi", 0),
        "buy_sent": entry_info.get("sent", 0),
        "buy_ncnt": entry_info.get("ncnt", 0),
        "sell_prob": sell_prob,
        "exit_type": exit_type,
    }
