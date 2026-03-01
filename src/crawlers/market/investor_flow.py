from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from ...common.config import KISAuth, KISConfig
from ...common.kis_client import KISClient


INVESTOR_ENDPOINT = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
INVESTOR_TR_ID = "FHPTJ04160001"

FIELDNAMES = [
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


@dataclass
class InvestorRow:
    date: str
    close: str
    high: str
    low: str
    prsn_shnu_vol: str
    prsn_shnu_tr_pbmn: str
    prsn_seln_vol: str
    prsn_seln_tr_pbmn: str
    frgn_shnu_vol: str
    frgn_shnu_tr_pbmn: str
    frgn_seln_vol: str
    frgn_seln_tr_pbmn: str
    orgn_shnu_vol: str
    orgn_shnu_tr_pbmn: str
    orgn_seln_vol: str
    orgn_seln_tr_pbmn: str


def _validate_constants():
    if not INVESTOR_ENDPOINT or not INVESTOR_TR_ID:
        raise ValueError("Set INVESTOR_ENDPOINT and INVESTOR_TR_ID before running.")


def _fetch_investor_rows(
    client: KISClient,
    stock_code: str,
    date_yyyymmdd: str,
    org_adj_prc: str = "0",
    etc_cls_code: str = "0",
) -> List[InvestorRow]:
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_INPUT_DATE_1": date_yyyymmdd,
        "FID_ORG_ADJ_PRC": org_adj_prc,
        "FID_ETC_CLS_CODE": etc_cls_code,
    }

    resp = client.get(
        url=f"{client.auth.config.api_url}{INVESTOR_ENDPOINT}",
        params=params,
        tr_id=INVESTOR_TR_ID,
    )
    if not resp:
        return []

    output = resp.get("output2", [])
    if not isinstance(output, list):
        output = [output]

    rows: List[InvestorRow] = []
    for item in output:
        rows.append(
            InvestorRow(
                date=(item.get("stck_bsop_date") or "").strip(),
                close=(item.get("stck_clpr") or "0").strip(),
                high=(item.get("stck_hgpr") or "0").strip(),
                low=(item.get("stck_lwpr") or "0").strip(),
                prsn_shnu_vol=(item.get("prsn_shnu_vol") or "0").strip(),
                prsn_shnu_tr_pbmn=(item.get("prsn_shnu_tr_pbmn") or "0").strip(),
                prsn_seln_vol=(item.get("prsn_seln_vol") or "0").strip(),
                prsn_seln_tr_pbmn=(item.get("prsn_seln_tr_pbmn") or "0").strip(),
                frgn_shnu_vol=(item.get("frgn_shnu_vol") or "0").strip(),
                frgn_shnu_tr_pbmn=(item.get("frgn_shnu_tr_pbmn") or "0").strip(),
                frgn_seln_vol=(item.get("frgn_seln_vol") or "0").strip(),
                frgn_seln_tr_pbmn=(item.get("frgn_seln_tr_pbmn") or "0").strip(),
                orgn_shnu_vol=(item.get("orgn_shnu_vol") or "0").strip(),
                orgn_shnu_tr_pbmn=(item.get("orgn_shnu_tr_pbmn") or "0").strip(),
                orgn_seln_vol=(item.get("orgn_seln_vol") or "0").strip(),
                orgn_seln_tr_pbmn=(item.get("orgn_seln_tr_pbmn") or "0").strip(),
            )
        )

    return rows


def _filter_by_date(rows: List[InvestorRow], start_date: str, end_date: str) -> List[InvestorRow]:
    return [r for r in rows if start_date <= r.date <= end_date]


def _iter_dates(start_yyyymmdd: str, end_yyyymmdd: str):
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def _row_to_dict(r: InvestorRow) -> dict:
    return {
        "날짜": r.date,
        "종가": r.close,
        "최고가": r.high,
        "최저가": r.low,
        "개인_매수2_거래량": r.prsn_shnu_vol,
        "개인_매수2_거래대금": r.prsn_shnu_tr_pbmn,
        "개인_매도_거래량": r.prsn_seln_vol,
        "개인_매도_거래대금": r.prsn_seln_tr_pbmn,
        "외국인_매수2_거래량": r.frgn_shnu_vol,
        "외국인_매수2_거래대금": r.frgn_shnu_tr_pbmn,
        "외국인_매도_거래량": r.frgn_seln_vol,
        "외국인_매도_거래대금": r.frgn_seln_tr_pbmn,
        "기관계_매수2_거래량": r.orgn_shnu_vol,
        "기관계_매수2_거래대금": r.orgn_shnu_tr_pbmn,
        "기관계_매도_거래량": r.orgn_seln_vol,
        "기관계_매도_거래대금": r.orgn_seln_tr_pbmn,
    }


def main():
    _validate_constants()

    p = argparse.ArgumentParser(description="Investor trade by stock (daily)")
    p.add_argument("--stock", default="005380", help="종목코드 (6자리)")
    p.add_argument("--start-date", default="20250101", help="YYYYMMDD")
    p.add_argument("--end-date", default="20260110", help="YYYYMMDD")
    p.add_argument("--repair", action="store_true", help="기존 CSV 기준 누락 날짜만 보완")
    args = p.parse_args()

    stock_code = args.stock.strip()
    start_date_str = args.start_date.strip()
    end_date_str = args.end_date.strip()

    config = KISConfig()
    if not config.validate():
        return

    auth = KISAuth(config)
    client = KISClient(auth)

    out_dir = Path("data/stock")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stock_code}.csv"

    by_date: dict[str, dict] = {}
    existing_dates = set()
    if args.repair and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = (row.get("날짜") or "").strip()
                if dt:
                    by_date[dt] = row
                    existing_dates.add(dt)

    dates_to_fetch = []
    for d in _iter_dates(start_date_str, end_date_str):
        if args.repair and d in existing_dates:
            continue
        dates_to_fetch.append(d)

    for d in dates_to_fetch:
        rows = _fetch_investor_rows(client, stock_code, d)
        rows = _filter_by_date(rows, start_date_str, end_date_str)
        for r in rows:
            if r.date:
                by_date[r.date] = _row_to_dict(r)

    ordered_dates = sorted(by_date.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for dt in ordered_dates:
            w.writerow(by_date[dt])

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
