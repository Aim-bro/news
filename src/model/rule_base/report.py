"""
=============================================================
report.py — Analysis & Report
=============================================================
Prints backtest results and saves CSV.
=============================================================
"""

import pandas as pd
try:
    from .config import *
except ImportError:
    from config import *


def print_summary(trades_df):
    if len(trades_df) == 0:
        print("No trades.")
        return

    wins = trades_df[trades_df["return_pct"] > 0]
    losses = trades_df[trades_df["return_pct"] <= 0]
    wr = len(wins) / len(trades_df) * 100

    print("=" * 60)
    print(f"  BACKTEST RESULT (BUY>={THRESHOLD_BUY*100:.0f}%, SELL>={THRESHOLD_SELL*100:.0f}%)")
    if TRAILING_STOP_PCT > 0:
        print(f"  Trailing Stop: {TRAILING_STOP_PCT*100:.0f}%")
    print("=" * 60)
    print(f"  Trades:      {len(trades_df)}")
    print(f"  Win Rate:    {wr:.1f}% ({len(wins)}W {len(losses)}L)")
    print(f"  Avg Return:  {trades_df['return_pct'].mean():+.2f}%")
    print(f"  Median:      {trades_df['return_pct'].median():+.2f}%")
    print(f"  Cumulative:  {trades_df['return_pct'].sum():+.1f}%")
    print(f"  Avg Hold:    {trades_df['hold_days'].mean():.0f} days")
    print(f"  Max Win:     {trades_df['return_pct'].max():+.1f}%")
    print(f"  Max Loss:    {trades_df['return_pct'].min():+.1f}%")

    if len(losses) > 0 and losses["return_pct"].mean() != 0:
        ratio = wins["return_pct"].mean() / abs(losses["return_pct"].mean())
        print(f"  Win/Loss:    {ratio:.2f}")

    # Exit type breakdown
    if "exit_type" in trades_df.columns:
        print(f"\n  Exit Types:")
        for et in trades_df["exit_type"].unique():
            sub = trades_df[trades_df["exit_type"] == et]
            print(f"    {et:16s} {len(sub):>3d} trades, "
                  f"WR {(sub['return_pct']>0).mean()*100:.0f}%, "
                  f"Avg {sub['return_pct'].mean():+.1f}%")

    print()


def print_by_stock(trades_df):
    if len(trades_df) == 0:
        return
    print(f"{'Stock':<16s} {'#':>3s} {'WinR':>6s} {'Avg':>7s} {'Total':>8s} {'Hold':>5s}")
    print("-" * 50)
    for name in sorted(trades_df["stock"].unique()):
        sub = trades_df[trades_df["stock"] == name]
        wr = (sub["return_pct"] > 0).mean() * 100
        print(f"  {name:<14s} {len(sub):>3d} {wr:>5.0f}% "
              f"{sub['return_pct'].mean():>+6.1f}% "
              f"{sub['return_pct'].sum():>+7.1f}% "
              f"{sub['hold_days'].mean():>4.0f}d")
    print()


def print_trades(trades_df, max_rows=50):
    if len(trades_df) == 0:
        return
    print(f"{'':>2s}{'Stock':<14s} {'Entry':>10s} {'Exit':>10s} {'Ret':>7s} "
          f"{'Hold':>5s} {'BuyP':>5s} {'SellP':>5s} {'Type':>8s}")
    print("-" * 72)
    for _, t in trades_df.head(max_rows).iterrows():
        mark = "+" if t["return_pct"] > 0 else "-"
        print(f"  {mark} {t['stock']:<12s} {t['entry_date']:>10s} {t['exit_date']:>10s} "
              f"{t['return_pct']:>+6.1f}% {t['hold_days']:>4d}d "
              f"{t['buy_prob']:>4.0f}% {t['sell_prob']:>4.0f}% "
              f"{t.get('exit_type',''):>8s}")
    if len(trades_df) > max_rows:
        print(f"  ... and {len(trades_df) - max_rows} more trades")
    print()


def save_csv(trades_df, path=OUTPUT_CSV):
    trades_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path} ({len(trades_df)} trades)")
