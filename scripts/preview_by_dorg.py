from pathlib import Path
import json
from collections import defaultdict

BASE_DIR = Path("data/news")

TARGET_DORGS = {
    "IRGO",
    # "코스닥 공시",
    # "머니투데이",
    # "이투데이",
}

MAX_PER_DORG = 20  # 언론사별 최대 출력 개수


def iter_items():
    for stock_dir in BASE_DIR.glob("*"):
        items_dir = stock_dir / "items"
        if not items_dir.is_dir():
            continue
        for p in items_dir.glob("*.json"):
            yield p


def main():
    bucket = defaultdict(list)

    for p in iter_items():
        item = json.loads(p.read_text(encoding="utf-8"))
        dorg = (item.get("dorg") or "").strip()
        if dorg in TARGET_DORGS:
            bucket[dorg].append(item)

    for dorg, items in bucket.items():
        print("\n" + "=" * 80)
        print(f"[{dorg}] 총 {len(items)}건 중 상위 {MAX_PER_DORG}건 미리보기")
        print("=" * 80)

        # 최신순 정렬
        items = sorted(
            items,
            key=lambda x: (x.get("data_dt", ""), x.get("data_tm", "")),
            reverse=True,
        )

        for it in items[:MAX_PER_DORG]:
            title = it.get("hts_pbnt_titl_cntt", "")
            dt = it.get("data_dt", "")
            tm = it.get("data_tm", "")
            codes = [it.get(f"iscd{i}") for i in range(1, 11) if it.get(f"iscd{i}")]
            print(f"- [{dt} {tm}] {title}")
            print(f"  종목: {', '.join(codes)}")


if __name__ == "__main__":
    main()
