import argparse
import time
from datetime import datetime, timedelta

from .config import KISConfig
from .fetch import KISNewsFetcher
from .news_store import NewsStore
from .save_csv import append_raw_row


def _build_news_title_params(
    *,
    stock_code: str,
    date_yyyymmdd: str,
    hour_hhmmss: str,
    keyword: str,
    input_srno: str,
):
    return {
        "FID_NEWS_OFER_ENTP_CODE": "",
        "FID_COND_MRKT_CLS_CODE": "",
        "FID_INPUT_ISCD": stock_code,
        "FID_TITL_CNTT": keyword,
        "FID_INPUT_DATE_1": date_yyyymmdd,
        "FID_INPUT_HOUR_1": hour_hhmmss,
        "FID_RANK_SORT_CLS_CODE": "",
        "FID_INPUT_SRNO": input_srno,
    }


def parse_args():
    today = datetime.now().strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="KIS news-title 수집기")
    p.add_argument("--mode", default="single", choices=["single", "daily"], help="single=1회 호출, daily=해당 날짜 전체 수집")
    p.add_argument("--stock", default="005930", help='종목코드 (전체 조회는 "ALL")')
    p.add_argument("--date", default=today, help="YYYYMMDD (기본: 오늘)")
    p.add_argument("--start-date", default="", help="YYYYMMDD (range start, optional)")
    p.add_argument("--end-date", default="", help="YYYYMMDD (range end, optional)")
    p.add_argument("--hour", default="000000", help="HHMMSS (기본: 000000)")
    p.add_argument("--keyword", default="", help="제목 키워드(기본: 빈값)")
    p.add_argument("--srno", default="0", help="조회 시작 순번(기본: 0)")

    p.add_argument("--raw-path", default="data/raw.csv", help="raw 저장 경로")
    p.add_argument("--no-raw", action="store_true", help="raw 저장 스킵")

    return p.parse_args()


def _normalize_stock_code(stock: str) -> str:
    s = stock.strip()
    if s.upper() == "ALL":
        return ""  # 전체 피드
    return s




def _iter_dates(start_yyyymmdd: str, end_yyyymmdd: str):
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    if end < start:
        raise ValueError("end_date is before start_date")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def _collect_daily_for_date(
    *,
    fetcher: KISNewsFetcher,
    store: NewsStore,
    config: KISConfig,
    stock_code: str,
    target_date: str,
    hour_hhmmss: str,
    keyword: str,
    raw_path: str,
    no_raw: bool,
    srno_start: str,
):
    endpoint = "/uapi/domestic-stock/v1/quotations/news-title"
    tr_id = "FHKST01011800"

    srno = srno_start or "0"
    seen_srno = set()
    total_written = 0
    total_skipped = 0

    stock_label = "ALL" if stock_code == "" else stock_code
    print(f"daily start: date={target_date}, stock={stock_label}")

    while True:
        if srno in seen_srno:
            print(f"srno repeated, stop (srno={srno})")
            break
        seen_srno.add(srno)

        params = _build_news_title_params(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=srno,
        )

        batch = fetcher.fetch_news_title(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=srno,
        )

        if batch is None:
            print("batch fetch failed, stop")
            break

        if len(batch) == 0:
            print("no more data (0)")
            break

        if not no_raw:
            append_raw_row(
                raw_csv_path=raw_path,
                base_url=config.api_url,
                endpoint=endpoint,
                tr_id=tr_id,
                params=params,
                status_code=None,
                response_obj=batch,
            )

        valid_items = []
        reached_prev_day = False
        for item in batch:
            dt = (item.get("data_dt") or "").strip()
            if dt and dt < target_date:
                reached_prev_day = True
                break
            valid_items.append(item)

        if valid_items:
            result = store.store_news_items(valid_items)
            total_written += int(result["written"])
            total_skipped += int(result["skipped"])
            print(
                f"batch saved (written={result['written']}, skipped={result['skipped']}, "
                f"stocks={len(result['stocks_touched'])}) | srno={srno}"
            )
        else:
            print(f"batch empty for target_date={target_date} | srno={srno}")

        if reached_prev_day:
            print("reached previous day, stop")
            break

        next_srno = (valid_items[-1].get("cntt_usiq_srno") or "").strip()
        if not next_srno:
            print("next srno missing, stop")
            break

        srno = next_srno

    print(f"daily done: total_written={total_written}, total_skipped={total_skipped}")
    return total_written, total_skipped


def main():
    args = parse_args()

    print("🚀 KIS 뉴스/공시 데이터 수집 시작")

    config = KISConfig()
    if not config.validate():
        print("❌ 설정 검증 실패. 프로그램 종료.")
        return

    fetcher = KISNewsFetcher(config)

    endpoint = "/uapi/domestic-stock/v1/quotations/news-title"
    tr_id = "FHKST01011800"

    stock_code = _normalize_stock_code(args.stock)
    target_date = args.date.strip()
    hour_hhmmss = args.hour.strip()
    keyword = args.keyword

    if args.mode == "single":
        input_srno = args.srno.strip()

        params = _build_news_title_params(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=input_srno,
        )

        news_data = fetcher.fetch_news_title(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=input_srno,
        )

        if news_data is None:
            print("❌ 뉴스 데이터 조회 실패.")
            print("🎉 데이터 수집 완료")
            return

        if not args.no_raw:
            append_raw_row(
                raw_csv_path=args.raw_path,
                base_url=config.api_url,
                endpoint=endpoint,
                tr_id=tr_id,
                params=params,
                status_code=None,
                response_obj=news_data,
            )

        if isinstance(news_data, list) and len(news_data) > 0:
            store = NewsStore()
            result = store.store_news_items(news_data)
            print(
                f"💾 뉴스 저장 완료 "
                f"(written={result['written']}, skipped={result['skipped']}, "
                f"stocks={len(result['stocks_touched'])})"
            )
        else:
            print("⚠️ 조회는 성공했지만 저장할 뉴스가 없습니다(0건)")

        print("🎉 데이터 수집 완료")
        return

    if args.start_date or args.end_date:
        start_date = (args.start_date or target_date).strip()
        end_date = (args.end_date or target_date).strip()
        try:
            t0 = time.time()
            all_written = 0
            all_skipped = 0
            store = NewsStore()
            for d in _iter_dates(start_date, end_date):
                w, s = _collect_daily_for_date(
                    fetcher=fetcher,
                    store=store,
                    config=config,
                    stock_code=stock_code,
                    target_date=d,
                    hour_hhmmss=hour_hhmmss,
                    keyword=keyword,
                    raw_path=args.raw_path,
                    no_raw=args.no_raw,
                    srno_start="0",
                )
                all_written += w
                all_skipped += s
            elapsed = time.time() - t0
            print(f"range done: total_written={all_written}, total_skipped={all_skipped}")
            print(f"range elapsed_sec={elapsed:.1f}")
        except ValueError as e:
            print(f"date range error: {e}")
        return

# ========== daily 모드: target_date 하루치 전체 수집 ==========
    srno = args.srno.strip() or "0"
    store = NewsStore()

    seen_srno = set()
    total_written = 0
    total_skipped = 0

    print(f"🗓️ daily 모드 시작: date={target_date}, stock={'ALL' if stock_code == '' else stock_code}")

    while True:
        if srno in seen_srno:
            print(f"🛑 srno가 반복되어 중단 (srno={srno})")
            break
        seen_srno.add(srno)

        params = _build_news_title_params(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=srno,
        )

        batch = fetcher.fetch_news_title(
            stock_code=stock_code,
            date_yyyymmdd=target_date,
            hour_hhmmss=hour_hhmmss,
            keyword=keyword,
            input_srno=srno,
        )

        if batch is None:
            print("❌ 배치 조회 실패로 중단")
            break

        if len(batch) == 0:
            print("🛑 더 이상 데이터 없음(0건)")
            break

        if not args.no_raw:
            append_raw_row(
                raw_csv_path=args.raw_path,
                base_url=config.api_url,
                endpoint=endpoint,
                tr_id=tr_id,
                params=params,
                status_code=None,
                response_obj=batch,
            )

        # 날짜 컷: target_date보다 이전이 섞이면 거기서 종료
        valid_items = []
        reached_prev_day = False
        for item in batch:
            dt = (item.get("data_dt") or "").strip()
            if dt and dt < target_date:
                reached_prev_day = True
                break
            valid_items.append(item)

        if valid_items:
            result = store.store_news_items(valid_items)
            total_written += int(result["written"])
            total_skipped += int(result["skipped"])
            print(
                f"📦 batch 저장 "
                f"(written={result['written']}, skipped={result['skipped']}, "
                f"stocks={len(result['stocks_touched'])}) | srno={srno}"
            )
        else:
            print(f"⚠️ batch는 받았지만 target_date({target_date}) 데이터가 없음 | srno={srno}")

        if reached_prev_day:
            print("🛑 기준 날짜 이전 뉴스 도달, 수집 종료")
            break

        # 다음 커서: 마지막 아이템의 cntt_usiq_srno
        next_srno = (valid_items[-1].get("cntt_usiq_srno") or "").strip()
        if not next_srno:
            print("🛑 다음 srno를 만들 수 없어 중단")
            break

        srno = next_srno

    print(f"✅ daily 완료: total_written={total_written}, total_skipped={total_skipped}")
    print("🎉 데이터 수집 완료")


if __name__ == "__main__":
    main()
