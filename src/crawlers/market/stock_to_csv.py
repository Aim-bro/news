import argparse
import csv
from pathlib import Path

SELECTED_COLUMNS = [
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


def _normalize_stock_code(code: str) -> str:
    s = (code or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    return f"{int(digits):06d}"


def _parse_stock_filter(args) -> set[str] | None:
    raw_values: list[str] = []
    if args.stock:
        raw_values.extend(args.stock)
    if args.stocks:
        raw_values.extend(args.stocks.split(","))
    codes = {_normalize_stock_code(v) for v in raw_values if v and v.strip()}
    return codes or None


def main():
    p = argparse.ArgumentParser(description="Merge selected stock CSV files under data/stock")
    p.add_argument("--stock", action="append", default=[], help="종목코드 (반복 가능)")
    p.add_argument("--stocks", default="", help="쉼표 구분 종목코드 목록")
    p.add_argument("--output", default="data/stock/merged_stock.csv", help="출력 CSV 경로")
    args = p.parse_args()

    base_dir = Path("data/stock")
    out_path = Path(args.output)
    stock_filter = _parse_stock_filter(args)

    if not base_dir.exists():
        print(f"missing: {base_dir}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = [fp for fp in base_dir.glob("*.csv") if fp.name.lower() not in {"all_news.csv", "index.csv"}]
    if stock_filter:
        csv_files = [fp for fp in csv_files if _normalize_stock_code(fp.stem) in stock_filter]

    if not csv_files:
        if stock_filter:
            print(f"no matching stock csv files found for: {', '.join(sorted(stock_filter))}")
        else:
            print("no stock csv files found under data/stock/")
        return

    fieldnames = ["stock_code", *SELECTED_COLUMNS]

    row_count = 0
    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in sorted(csv_files):
            stock_code = _normalize_stock_code(fp.stem)
            with fp.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_out = {"stock_code": stock_code}
                    for col in SELECTED_COLUMNS:
                        row_out[col] = row.get(col, "")
                    writer.writerow(row_out)
                    row_count += 1

    if stock_filter:
        print(f"saved: {out_path} ({row_count} rows, stocks: {', '.join(sorted(stock_filter))})")
    else:
        print(f"saved: {out_path} ({row_count} rows)")


if __name__ == "__main__":
    main()
