from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from .config import KISAuth, KISConfig
from .kis_client import KISClient



INVESTOR_ENDPOINT = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
INVESTOR_TR_ID = "FHPTJ04160001"


@dataclass
class InvestorRow:
    date: str
    close: str
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


def main():
    _validate_constants()

    stock_code = "005380"
    start_date_str = "20250101"
    end_date_str = "20260110"

    config = KISConfig()
    if not config.validate():
        return

    auth = KISAuth(config)
    client = KISClient(auth)

    by_date = {}
    for d in _iter_dates(start_date_str, end_date_str):
        rows = _fetch_investor_rows(client, stock_code, d)
        rows = _filter_by_date(rows, start_date_str, end_date_str)
        for r in rows:
            if r.date and r.date not in by_date:
                by_date[r.date] = r

    rows = [by_date[k] for k in sorted(by_date.keys())]

    out_dir = Path("data/stock")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stock_code}.csv"
    fieldnames = list(InvestorRow.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
