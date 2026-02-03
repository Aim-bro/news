import csv
from pathlib import Path


def main():
    base_dir = Path("data/news")
    out_path = Path("data/stock/all_news.csv")

    if not base_dir.exists():
        print(f"missing: {base_dir}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    index_files = list(base_dir.glob("*/index.csv"))
    if not index_files:
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
            stock_code = fp.parent.name
            with fp.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_out = {"stock_code": stock_code}
                    row_out.update(row)
                    writer.writerow(row_out)

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
