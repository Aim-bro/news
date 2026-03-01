"""
=============================================================
config.py — All Tunable Parameters
=============================================================
Edit this file to customize the model behavior.
Run: python main.py
=============================================================
"""

# --- Data ---
# news project layout: merged model features are stored under data/
DATA_PATH = "data/features_with_news.csv"

# --- Core Parameters ---
LOOKBACK = 20       # Days to look back for min/max detection
DELAY = 3           # Confirm after 3 days (t-3 = target date)

# --- Thresholds (0.0 ~ 1.0) ---
THRESHOLD_BUY = 0.85    # Low probability >= this → BUY
THRESHOLD_SELL = 0.85   # High probability >= this → SELL

# --- Category Weights ---
# Price Structure (max ~35%)
W_LEFT_EXTREME = 0.15       # Target is N-day high/low
W_RIGHT_EXTREME = 0.10      # Still extreme after 3 days
W_MOVE_STRONG = 0.10        # 5%+ move in 3 days
W_MOVE_MILD = 0.05          # 2%+ move in 3 days

# Technical Indicators (max ~30%)
W_RSI_EXTREME = 0.12        # RSI >= 75 or <= 25
W_RSI_STRONG = 0.08         # RSI >= 70 or <= 30
W_RSI_MILD = 0.04           # RSI >= 65 or <= 35
W_GAP_EXTREME = 0.10        # Disparity >= 115 or <= 85
W_GAP_STRONG = 0.07         # Disparity >= 110 or <= 90
W_GAP_MILD = 0.04           # Disparity >= 105 or <= 95
W_BB_EXTREME = 0.08         # BB%B >= 1.05 or <= -0.05
W_BB_STRONG = 0.04          # BB%B >= 0.95 or <= 0.05

# Supply/Demand Flip (max ~35%)
W_INST_STRONG = 0.12        # Institutional flip > 30
W_INST_MID = 0.07           # Institutional flip > 10
W_INST_MILD = 0.03          # Institutional flip > 3
W_FOR_STRONG = 0.10         # Foreign flip > 30
W_FOR_MID = 0.06            # Foreign flip > 10
W_FOR_MILD = 0.03           # Foreign flip > 3
W_RET_STRONG = 0.08         # Retail surge > 30 (contrarian)
W_RET_MID = 0.04            # Retail surge > 10

# Combo Bonus (max ~10%)
W_COMBO_RSI_INST = 0.05     # RSI extreme + institutional flip
W_COMBO_PRICE_INST = 0.05   # Price extreme + decline + institutional flip

# --- News Sentiment (Bonus/Penalty on top of base score) ---
NEWS_BONUS_STRONG = 0.08    # Sentiment aligns strongly (|sent| >= 0.3)
NEWS_BONUS_MILD = 0.04      # Sentiment aligns mildly (|sent| >= 0.1)
NEWS_BONUS_MOM_STRONG = 0.05  # Sentiment momentum aligns (|mom| >= 0.3)
NEWS_BONUS_MOM_MILD = 0.02    # Sentiment momentum mild (|mom| >= 0.1)
NEWS_PENALTY_STRONG = 0.06  # Sentiment opposes strongly
NEWS_PENALTY_MILD = 0.03    # Sentiment opposes mildly

# --- Market Regime / Relative Strength (Bonus/Penalty) ---
# relative_strength_5: stock 5d return - index 5d return
REGIME_RS_BONUS_STRONG = 0.04
REGIME_RS_BONUS_MILD = 0.02
REGIME_RS_PENALTY_STRONG = 0.04
REGIME_RS_PENALTY_MILD = 0.02

# index_ret_5: 5-day benchmark index return (KOSPI)
REGIME_INDEX_BONUS_STRONG = 0.04
REGIME_INDEX_BONUS_MILD = 0.02
REGIME_INDEX_PENALTY_STRONG = 0.04
REGIME_INDEX_PENALTY_MILD = 0.02

# volume_z: 60-day z-score of approx_total_amt (eventfulness/capitulation/blowoff proxy)
REGIME_VOLZ_BONUS_STRONG = 0.03
REGIME_VOLZ_BONUS_MILD = 0.015
REGIME_VOLZ_PENALTY_LOW_LIQ = 0.015

# --- Trailing Stop (set to 0 to disable) ---
TRAILING_STOP_PCT = 0.0     # e.g. 0.15 = sell if price drops 15% from peak

# --- Output ---
OUTPUT_CSV = "backtest_result.csv"
