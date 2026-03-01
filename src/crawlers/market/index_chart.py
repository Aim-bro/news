from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from ...common.config import KISAuth, KISConfig
from ...common.kis_client import KISClient


INDEX_ENDPOINT = "/uapi/domestic-stock/v1/quotations/inquire-index-daily-price"
INDEX_TR_ID = "FHPUP02120000"

TARGET_CODES = ["0001", "1001"]  # KOSPI, KOSDAQ
INDEX_NAME_BY_CODE = {
    "0001": "코스피",
    "1001": "코스닥",
}
MARKET_DIV_CODE = "U"
PERIOD_DIV_CODE = "D"

FIELDNAMES = [
    "지수코드",
    "지수명",
    "날짜",
    "종가",
    "시가",
    "고가",
    "저가",
]


@dataclass
class IndexRow:
    index_code: str
    index_name: str
    date: str
    close: str
    open: str
    high: str
    low: str


def _iter_dates(start_yyyymmdd: str, end_yyyymmdd: str):
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    if end < start:
        raise ValueError("end-date is earlier than start-date")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def _fetch_index_rows_for_day(
    client: KISClient,
    index_code: str,
    date_yyyymmdd: str,
) -> List[IndexRow]:
    params = {
        "FID_COND_MRKT_DIV_CODE": MARKET_DIV_CODE,  # always U
        "FID_INPUT_ISCD": index_code,               # 0001 / 1001
        "FID_PERIOD_DIV_CODE": PERIOD_DIV_CODE,     # always D
        "FID_INPUT_DATE_1": date_yyyymmdd,          # 하루씩 순회
    }

    resp = client.get(
        url=f"{client.auth.config.api_url}{INDEX_ENDPOINT}",
        params=params,
        tr_id=INDEX_TR_ID,
    )
    if not resp:
        return []

    output = resp.get("output2", [])
    if not isinstance(output, list):
        output = [output]

    rows: List[IndexRow] = []
    for item in output:
        rows.append(
            IndexRow(
                index_code=index_code,
                index_name=INDEX_NAME_BY_CODE.get(index_code, index_code),
                date=(item.get("stck_bsop_date") or "").strip(),
                close=(item.get("bstp_nmix_prpr") or "0").strip(),
                open=(item.get("bstp_nmix_oprc") or "0").strip(),
                high=(item.get("bstp_nmix_hgpr") or "0").strip(),
                low=(item.get("bstp_nmix_lwpr") or "0").strip(),
            )
        )
    return rows


def _row_to_dict(r: IndexRow) -> dict:
    return {
        "지수코드": r.index_code,
        "지수명": r.index_name,
        "날짜": r.date,
        "종가": r.close,
        "시가": r.open,
        "고가": r.high,
        "저가": r.low,
    }


def _save_rows(out_path: Path, rows: List[IndexRow], *, repair: bool):
    by_key: dict[tuple[str, str], dict] = {}
    existing_keys: set[tuple[str, str]] = set()

    if repair and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("지수코드") or "").strip()
                dt = (row.get("날짜") or "").strip()
                if code and dt:
                    key = (code, dt)
                    by_key[key] = row
                    existing_keys.add(key)

    for r in rows:
        if not r.date:
            continue
        key = (r.index_code, r.date)
        if repair and key in existing_keys:
            continue
        by_key[key] = _row_to_dict(r)

    ordered_keys = sorted(by_key.keys(), key=lambda x: (x[1], x[0]))  # date, code
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for key in ordered_keys:
            w.writerow(by_key[key])


def main():
    p = argparse.ArgumentParser(description="KIS index daily price crawler (KOSPI/KOSDAQ)")
    p.add_argument("--start-date", required=True, help="YYYYMMDD")
    p.add_argument("--end-date", required=True, help="YYYYMMDD")
    p.add_argument("--repair", action="store_true", help="기존 index.csv 기준 누락만 추가")
    p.add_argument(
        "--output",
        default="data/index.csv",
        help="출력 CSV 경로 (기본: data/index.csv)",
    )
    args = p.parse_args()

    config = KISConfig()
    if not config.validate():
        return

    auth = KISAuth(config)
    client = KISClient(auth)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[tuple[str, str]] = set()
    if args.repair and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("지수코드") or "").strip()
                dt = (row.get("날짜") or "").strip()
                if code and dt:
                    existing_keys.add((code, dt))

    fetched_rows: List[IndexRow] = []
    start_date = args.start_date.strip()
    end_date = args.end_date.strip()

    for index_code in TARGET_CODES:
        print(f"[index={index_code}] crawling {start_date} ~ {end_date}")
        for d in _iter_dates(start_date, end_date):
            if args.repair and (index_code, d) in existing_keys:
                continue
            day_rows = _fetch_index_rows_for_day(
                client=client,
                index_code=index_code,
                date_yyyymmdd=d,
            )
            for r in day_rows:
                if start_date <= r.date <= end_date:
                    fetched_rows.append(r)

    _save_rows(out_path, fetched_rows, repair=args.repair)
    print(f"saved: {out_path} (+{len(fetched_rows)} fetched)")


if __name__ == "__main__":
    main()
