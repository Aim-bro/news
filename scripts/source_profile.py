from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BASE_DIR = Path("data/news")


def iter_item_files(base_dir: Path) -> Iterable[Path]:
    for stock_dir in base_dir.glob("*"):
        items_dir = stock_dir / "items"
        if items_dir.is_dir():
            yield from items_dir.glob("*.json")


def load_item(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def count_related_codes(item: Dict) -> int:
    n = 0
    for i in range(1, 11):
        if (item.get(f"iscd{i}") or "").strip():
            n += 1
    return n


@dataclass
class SourceStats:
    count: int = 0
    related_cnt_sum: int = 0
    title_counter: Counter = None
    template_like: int = 0  # 템플릿성(노이즈) 추정

    def __post_init__(self):
        if self.title_counter is None:
            self.title_counter = Counter()


TEMPLATE_KEYWORDS = [
    "인기검색",
    "테마동향",
    "기술적 분석",
    "매수체결 상위",
    "오전장",
    "특징주",
]


def is_template_title(title: str) -> bool:
    t = (title or "").strip()
    return any(k in t for k in TEMPLATE_KEYWORDS)


def main():
    if not BASE_DIR.exists():
        print(f"❌ {BASE_DIR} 가 없습니다. 먼저 수집을 수행하세요")
        return

    stats: Dict[Tuple[str, str], SourceStats] = defaultdict(SourceStats)
    dates = Counter()

    for fp in iter_item_files(BASE_DIR):
        item = load_item(fp)

        dorg = (item.get("dorg") or "").strip()
        code = (item.get("news_ofer_entp_code") or "").strip()
        dt = (item.get("data_dt") or "").strip()
        title = (item.get("hts_pbnt_titl_cntt") or "").strip()

        key = (dorg, code)
        s = stats[key]
        s.count += 1
        s.related_cnt_sum += count_related_codes(item)
        s.title_counter[title] += 1
        if is_template_title(title):
            s.template_like += 1

        if dt:
            dates[dt] += 1

    # 출력: 상위 30개 공급자
    rows = []
    for (dorg, code), s in stats.items():
        dup_rate = 0.0
        if s.count > 0:
            dup_rate = 1.0 - (len(s.title_counter) / s.count)  # 0에 가까울수록 중복 적음
        rel_avg = (s.related_cnt_sum / s.count) if s.count else 0.0
        template_rate = (s.template_like / s.count) if s.count else 0.0

        rows.append((s.count, dorg, code, rel_avg, dup_rate, template_rate))

    rows.sort(reverse=True)

    print("=== 수집 날짜 분포(상위) ===")
    for dt, c in dates.most_common(10):
        print(dt, c)

    print("\n=== 언론사/공급자 프로파일 (상위 30) ===")
    print("count | dorg | code | related_avg | dup_rate | template_rate")
    for count, dorg, code, rel_avg, dup_rate, template_rate in rows[:30]:
        print(
            f"{count:5d} | {dorg[:12]:12s} | {code:>2s} | "
            f"{rel_avg:10.2f} | {dup_rate:8.2%} | {template_rate:12.2%}"
        )

    print("\nTip: count 높고 dup/template 낮고 related_avg 높은 공급자를 우선 후보로 잡으면 좋음")


if __name__ == "__main__":
    main()
