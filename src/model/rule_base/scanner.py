"""
=============================================================
scanner.py — Daily Signal Scanner
=============================================================
Scans all stocks and prints today's inflection signals.
Use this for daily monitoring after market close.

Usage:
    python scanner.py                    # scan all stocks
    python scanner.py --stock 삼성SDI     # scan one stock
    python scanner.py --threshold 75     # custom threshold
    python scanner.py --top 10           # show top N signals
=============================================================
"""

import pandas as pd
import numpy as np
import argparse
try:
    from .config import *
    from .probability import calc_probability
except ImportError:
    from config import *
    from probability import calc_probability


def scan(features_df, threshold=0.50, stock_filter=None, top_n=None):
    """
    Scan latest signals for all stocks.
    Lower threshold than backtest to show more candidates.
    """
    results = []

    for code, group in features_df.groupby("종목코드"):
        name = group["종목명"].iloc[0]
        if stock_filter and stock_filter not in name:
            continue

        g = group.sort_values("날짜").reset_index(drop=True)
        dates = g["날짜"].dt.strftime("%Y-%m-%d").values
        n = len(g)

        # Only check latest date
        today = n - 1
        target = today - DELAY

        if target < LOOKBACK:
            continue

        r = calc_probability(g, target)
        if r is None:
            continue

        hp = r["high_prob"] * 100
        lp = r["low_prob"] * 100

        if hp >= threshold * 100 or lp >= threshold * 100:
            signal = "LOW" if lp > hp else "HIGH"
            prob = max(hp, lp)
            results.append({
                "stock": name,
                "today": dates[today],
                "target": dates[target],
                "signal": signal,
                "prob": round(prob, 1),
                "high%": round(hp, 1),
                "low%": round(lp, 1),
                "base_h": round(r["base_high"] * 100, 1),
                "base_l": round(r["base_low"] * 100, 1),
                "news_h": round(r["news_bonus_h"] * 100, 1),
                "news_l": round(r["news_bonus_l"] * 100, 1),
                "rsi": r["rsi"],
                "gap": r["gap20"],
                "inst": r["inst_flip"],
                "fore": r["for_flip"],
                "sent": r["sentiment"],
                "ncnt": r["news_count"],
                "price": int(r["target_price"]),
            })

    results.sort(key=lambda x: x["prob"], reverse=True)
    if top_n:
        results = results[:top_n]

    return results


def print_scan(results):
    if not results:
        print("No signals found.")
        return

    print(f"\n{'='*80}")
    print(f"  INFLECTION SIGNAL SCAN — {results[0]['today']}")
    print(f"{'='*80}")
    print(f"  {'Stock':<14s} {'Signal':>6s} {'Prob':>5s} {'Base':>5s} {'News':>5s} "
          f"{'RSI':>5s} {'Gap':>5s} {'Inst':>6s} {'Fore':>6s} {'Sent':>5s} {'#N':>3s}")
    print(f"  {'-'*74}")

    for r in results:
        color_prob = r["prob"]
        base = r["base_l"] if r["signal"] == "LOW" else r["base_h"]
        news = r["news_l"] if r["signal"] == "LOW" else r["news_h"]
        sig_mark = "▼LOW" if r["signal"] == "LOW" else "▲HIGH"
        alert = " ◀◀◀" if r["prob"] >= THRESHOLD_BUY * 100 else (" ◀" if r["prob"] >= 70 else "")

        print(f"  {r['stock']:<14s} {sig_mark:>6s} {r['prob']:>4.0f}% "
              f"{base:>4.0f}% {news:>+4.0f}% "
              f"{r['rsi']:>5.1f} {r['gap']:>5.1f} "
              f"{r['inst']:>+5.0f} {r['fore']:>+5.0f} "
              f"{r['sent']:>+.2f} {r['ncnt']:>3d}{alert}")

    buy_signals = [r for r in results if r["signal"] == "LOW" and r["prob"] >= THRESHOLD_BUY * 100]
    sell_signals = [r for r in results if r["signal"] == "HIGH" and r["prob"] >= THRESHOLD_SELL * 100]

    if buy_signals:
        print(f"\n  ** BUY ALERT (>={THRESHOLD_BUY*100:.0f}%): {', '.join(r['stock'] for r in buy_signals)}")
    if sell_signals:
        print(f"  ** SELL ALERT (>={THRESHOLD_SELL*100:.0f}%): {', '.join(r['stock'] for r in sell_signals)}")
    if not buy_signals and not sell_signals:
        print(f"\n  No signals above threshold ({THRESHOLD_BUY*100:.0f}%).")
    print()


def scan_history(features_df, stock_name, last_n=20, threshold=0.30):
    """
    Show probability history for one stock (last N trading days).
    """
    group = features_df[features_df["종목명"] == stock_name]
    if len(group) == 0:
        print(f"Stock '{stock_name}' not found.")
        return

    g = group.sort_values("날짜").reset_index(drop=True)
    dates = g["날짜"].dt.strftime("%Y-%m-%d").values
    n = len(g)

    print(f"\n{'='*80}")
    print(f"  {stock_name} — Probability History (last {last_n} days)")
    print(f"{'='*80}")
    print(f"  {'Today':>10s} {'Target':>10s} {'Price':>8s} {'High%':>6s} {'Low%':>6s} "
          f"{'RSI':>5s} {'Gap':>5s} {'Inst':>6s} {'Sent':>5s} {'Signal':>8s}")
    print(f"  {'-'*72}")

    for today in range(max(LOOKBACK + DELAY + 2, n - last_n), n):
        target = today - DELAY
        r = calc_probability(g, target)
        if r is None:
            continue

        hp = r["high_prob"] * 100
        lp = r["low_prob"] * 100
        sig = ""
        if hp >= THRESHOLD_SELL * 100:
            sig = "◀ SELL"
        elif lp >= THRESHOLD_BUY * 100:
            sig = "◀ BUY"
        elif hp >= 70:
            sig = "~ high"
        elif lp >= 70:
            sig = "~ low"

        print(f"  {dates[today]:>10s} {dates[target]:>10s} "
              f"{int(r['target_price']):>8,} "
              f"{hp:>5.1f}% {lp:>5.1f}% "
              f"{r['rsi']:>5.1f} {r['gap20']:>5.1f} "
              f"{r['inst_flip']:>+5.0f} {r['sentiment']:>+.2f} "
              f"{sig:>8s}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Daily Inflection Signal Scanner")
    parser.add_argument("--stock", type=str, help="Filter by stock name")
    parser.add_argument("--threshold", type=float, default=50, help="Min probability %% (default: 50)")
    parser.add_argument("--top", type=int, help="Show top N signals only")
    parser.add_argument("--history", type=str, help="Show probability history for a stock")
    parser.add_argument("--days", type=int, default=20, help="History days (default: 20)")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["날짜"] = pd.to_datetime(df["날짜"])
    print(f"Loaded: {len(df):,} rows, {df['종목명'].nunique()} stocks")

    if args.history:
        scan_history(df, args.history, last_n=args.days)
    else:
        results = scan(df, threshold=args.threshold / 100,
                       stock_filter=args.stock, top_n=args.top)
        print_scan(results)


if __name__ == "__main__":
    main()
