import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

# === 언론사 allowlist ===
ALLOW_DORG = {
    "뉴스핌",
    "서울경제",
    "한국경제신문",
    "머니투데이",
    "연합뉴스",
    "이데일리",
    "이투데이",
}


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _extract_related_codes(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    item에서 관련 종목코드/종목명(최대 10개) 추출
    반환: [(iscd, kor_name), ...]
    """
    out: List[Tuple[str, str]] = []
    for i in range(1, 11):
        code = (item.get(f"iscd{i}") or "").strip()
        name = (item.get(f"kor_isnm{i}") or "").strip()
        if code:
            out.append((code, name))
    return out


@dataclass
class NewsStore:
    base_dir: Path = Path("data/news")

    def _stock_dir(self, stock_code: str) -> Path:
        return self.base_dir / stock_code

    def _items_dir(self, stock_code: str) -> Path:
        return self._stock_dir(stock_code) / "items"

    def _seen_path(self, stock_code: str) -> Path:
        return self._stock_dir(stock_code) / "seen_ids.txt"

    def _index_path(self, stock_code: str) -> Path:
        return self._stock_dir(stock_code) / "index.csv"

    def _load_seen(self, stock_code: str) -> Set[str]:
        p = self._seen_path(stock_code)
        if not p.exists():
            return set()
        return set(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())

    def _append_seen(self, stock_code: str, new_ids: Iterable[str]):
        p = self._seen_path(stock_code)
        _safe_mkdir(p.parent)
        with p.open("a", encoding="utf-8") as f:
            for cid in new_ids:
                f.write(f"{cid}\n")

    def _ensure_index_header(self, stock_code: str):
        p = self._index_path(stock_code)
        _safe_mkdir(p.parent)
        if p.exists():
            return
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ingested_at",
                    "content_id",
                    "published_dt",
                    "published_tm",
                    "published_at",
                    "title",
                    "source_name",
                    "source_code",
                    "category_code",
                    "primary_stock_code",
                    "primary_stock_name",
                ]
            )

    def _write_item_json(self, stock_code: str, content_id: str, item: Dict[str, Any]):
        items_dir = self._items_dir(stock_code)
        _safe_mkdir(items_dir)
        fp = items_dir / f"{content_id}.json"
        if fp.exists():
            return False
        # atomic write
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fp)
        return True

    def store_news_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        items(list): news-title output 리스트

        동작
        - 각 item을 관련 종목 코드들에 fan-out 저장
        - 종목 폴더별로 content_id 중복 제거
        - index.csv에 요약 레코드 append
        """
        _safe_mkdir(self.base_dir)

        total_written = 0
        total_skipped = 0
        stocks_touched: Set[str] = set()

        # 종목별 seen 캐시는 필요할 때만 로드
        seen_cache: Dict[str, Set[str]] = {}
        

        for item in items:
            
            source_name = (item.get("dorg") or "").strip()
            if source_name not in ALLOW_DORG:
                total_skipped += 1
                continue

            
            content_id = (item.get("cntt_usiq_srno") or "").strip()
            if not content_id:
                # content_id 없는 건 운영에서 다루기 애매해서 스킵
                total_skipped += 1
                continue

            related = _extract_related_codes(item)
            if not related:
                total_skipped += 1
                continue

            published_dt = (item.get("data_dt") or "").strip()
            published_tm = (item.get("data_tm") or "").strip()
            title = (item.get("hts_pbnt_titl_cntt") or "").strip()
            source_code = (item.get("news_ofer_entp_code") or "").strip()
            category_code = (item.get("news_lrdv_code") or "").strip()
            source_name = (item.get("dorg") or "").strip()

            # 대표 종목은 iscd1/kor_isnm1로 잡기
            primary_code, primary_name = related[0]

            # fan-out: 관련 종목 각각에 저장
            for stock_code, stock_name in related:
                stocks_touched.add(stock_code)
                if stock_code not in seen_cache:
                    seen_cache[stock_code] = self._load_seen(stock_code)

                if content_id in seen_cache[stock_code]:
                    total_skipped += 1
                    continue

                wrote = self._write_item_json(stock_code, content_id, item)
                if not wrote:
                    # 파일이 이미 있으면 seen도 있다고 간주
                    seen_cache[stock_code].add(content_id)
                    total_skipped += 1
                    continue

                # index.csv append
                self._ensure_index_header(stock_code)
                idx_path = self._index_path(stock_code)
                with idx_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    published_at = ""
                    if published_dt and published_tm and len(published_dt) == 8 and len(published_tm) == 6:
                        published_at = f"{published_dt}T{published_tm}"
                    w.writerow(
                        [
                            _now_iso(),
                            content_id,
                            published_dt,
                            published_tm,
                            published_at,
                            title,
                            source_name,
                            source_code,
                            category_code,
                            primary_code,
                            primary_name,
                        ]
                    )

                # seen 업데이트 (메모리 + 파일)
                seen_cache[stock_code].add(content_id)
                self._append_seen(stock_code, [content_id])

                total_written += 1

        return {
            "written": total_written,
            "skipped": total_skipped,
            "stocks_touched": sorted(stocks_touched),
        }
