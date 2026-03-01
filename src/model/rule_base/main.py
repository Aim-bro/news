"""
=============================================================
main.py — Run Full Backtest
=============================================================
Usage:
    python main.py                         # default settings
    python main.py --buy 75 --sell 85      # custom thresholds
    python main.py --stop 15               # trailing stop 15%
    python main.py --period 2025-01-01     # filter start date
=============================================================
"""

import pandas as pd
import argparse
try:
    from .config import *
    from .backtest import run_backtest
    from .report import print_summary, print_by_stock, print_trades, save_csv
except ImportError:
    from config import *
    from backtest import run_backtest
    from report import print_summary, print_by_stock, print_trades, save_csv


def main():
    parser = argparse.ArgumentParser(description="Inflection Point Backtest")
    parser.add_argument("--buy", type=float, default=THRESHOLD_BUY * 100,
                        help="Buy threshold %% (default: from config)")
    parser.add_argument("--sell", type=float, default=THRESHOLD_SELL * 100,
                        help="Sell threshold %% (default: from config)")
    parser.add_argument("--stop", type=float, default=TRAILING_STOP_PCT * 100,
                        help="Trailing stop %% (0=disabled, default: from config)")
    parser.add_argument("--period", type=str, default=None,
                        help="Start date filter (e.g. 2025-01-01)")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="Output CSV path")
    args = parser.parse_args()

    buy_thr = args.buy / 100
    sell_thr = args.sell / 100
    stop_pct = args.stop / 100

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["날짜"] = pd.to_datetime(df["날짜"])

    if args.period:
        df = df[df["날짜"] >= args.period]

    print(f"Data: {len(df):,} rows | {df['종목명'].nunique()} stocks | "
          f"{df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}")
    print(f"Settings: BUY>={buy_thr*100:.0f}% SELL>={sell_thr*100:.0f}% "
          f"STOP={stop_pct*100:.0f}% LOOKBACK={LOOKBACK} DELAY={DELAY}\n")

    trades, signals = run_backtest(df, buy_thr, sell_thr, stop_pct)

    print_summary(trades)
    print_by_stock(trades)
    print_trades(trades)
    save_csv(trades, args.output)


if __name__ == "__main__":
    main()
