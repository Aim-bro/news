# Inflection Point Probability Model (Rule-Based V3)

Detects stock price inflection points using price structure, technical indicators,
supply/demand shifts, and news sentiment.

## Quick Start

```bash
# 0. Preprocess raw data (first time only)
python preprocess.py --price 23stocks.csv --news df_filtered.csv
# → creates features_with_news.csv

# 1. Or place pre-built data file
cp features_with_news.csv ./   # or set DATA_PATH in config.py

# 2. Run backtest (default: 85% threshold)
python main.py

# 3. Custom thresholds
python main.py --buy 75 --sell 85

# 4. With trailing stop
python main.py --buy 75 --sell 85 --stop 15

# 5. News period only
python main.py --period 2025-01-01

# 6. Daily signal scan
python scanner.py
python scanner.py --stock 삼성SDI
python scanner.py --threshold 60 --top 10

# 7. Single stock probability history
python scanner.py --history 네이버 --days 30
```

## Files

| File | Purpose |
|------|---------|
| `preprocess.py` | **Raw data → features CSV** (technicals + sentiment) |
| `config.py` | All tunable parameters (thresholds, weights, etc.) |
| `probability.py` | Core engine: calculates P(high) and P(low) |
| `backtest.py` | Simulates trades with buy/sell/trailing stop |
| `report.py` | Prints results and saves CSV |
| `scanner.py` | Daily signal scanner + stock history viewer |
| `main.py` | Entry point for full backtest |

## Tunable Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `THRESHOLD_BUY` | 0.85 | Low prob >= this triggers BUY |
| `THRESHOLD_SELL` | 0.85 | High prob >= this triggers SELL |
| `LOOKBACK` | 20 | Days to compare for min/max |
| `DELAY` | 3 | Days to wait before confirming |
| `TRAILING_STOP_PCT` | 0.0 | Trailing stop (0=off, 0.15=15%) |
| `W_*` | various | Category weights (see config.py) |
| `NEWS_BONUS_*` | various | News sentiment bonus/penalty |

## Requirements

```
pandas
numpy
```

## Data Format

CSV with columns (Korean):
- 종목코드, 날짜, 종가, 종목명
- RSI_14, 이격도_20, BB_PctB
- 기관계_순매수, 외국인_순매수, 개인_순매수
- (optional) news_count, sent_mean, sent_momentum
