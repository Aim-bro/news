# Model Workspace

`src/model` is the top-level modeling workspace.

## Current Structure

- `preprocess.py`: builds `data/features_with_news.csv`
- `rule_base/`: rule-based inflection model (probability/backtest/scanner/config)
- future models (e.g. `lgbm/`) can be added here

## Commands

Build features:

```bash
python src/model/preprocess.py --price data/stock/merged_stock.csv --news data/stock/all_news.csv --index data/index.csv --output data/features_with_news.csv
```

Rule-based backtest:

```bash
python -m src.model.rule_base.main
```

Rule-based scanner:

```bash
python -m src.model.rule_base.scanner --top 20
```

