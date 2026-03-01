import argparse
import csv
from pathlib import Path


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
    p = argparse.ArgumentParser(description="Merge stored news index CSVs")
    p.add_argument("--stock", action="append", default=[], help="종목코드 (반복 가능)")
    p.add_argument("--stocks", default="", help="쉼표 구분 종목코드 목록")
    p.add_argument("--output", default="data/stock/all_news.csv", help="출력 CSV 경로")
    args = p.parse_args()

    base_dir = Path("data/news")
    out_path = Path(args.output)
    stock_filter = _parse_stock_filter(args)

    if not base_dir.exists():
        print(f"missing: {base_dir}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    index_files = list(base_dir.glob("*/index.csv"))
    if stock_filter:
        index_files = [
            fp for fp in index_files
            if _normalize_stock_code(fp.parent.name) in stock_filter
        ]
    if not index_files:
        if stock_filter:
            print(f"no matching index.csv files found for: {', '.join(sorted(stock_filter))}")
        else:
            print("no index.csv files found under data/news/")
        return

    # Collect headers from all index.csv files
    fieldnames = ["stock_code"]
    seen = set(fieldnames)
    for fp in index_files:
        with fp.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            for h in header:
                if h not in seen:
                    seen.add(h)
                    fieldnames.append(h)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in index_files:
            stock_code = _normalize_stock_code(fp.parent.name)
            with fp.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_out = {"stock_code": stock_code}
                    row_out.update(row)
                    writer.writerow(row_out)

    if stock_filter:
        print(f"saved: {out_path} (stocks: {', '.join(sorted(stock_filter))})")
    else:
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
