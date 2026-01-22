from .config import KISConfig
from .fetch import KISNewsFetcher
from .save_csv import KISCSVSaver, append_raw_row

from datetime import datetime

def main():
    """메인 실행 함수"""
    print("🚀 KIS 뉴스/공시 데이터 수집 시작")
    
    # 설정 로드
    config = KISConfig()
    if not config.validate():
        print("❌ 설정 검증 실패. 프로그램 종료.")
        return
    
    # 컴포넌트 인스턴스 생성
    fetcher = KISNewsFetcher(config)
    saver = KISCSVSaver()
    
    # 뉴스 데이터 조회
    news_data = fetcher.fetch_news()
    


    if news_data:
        # 원본 데이터 저장
        # fetcher 내부 값을 최대한 가져오고, 없으면 네가 적어둔 값으로 fallback
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": "0001",
            "fid_blng_cls_code": "0",
            "fid_trtm_cl_code": "111111",
            "fid_div_cls_code": "0",
        }

        endpoint = "/uapi/domestic-stock/v1/quotations/market-comprehensive"
        tr_id = "FHPST01010400"  # fetch.py에서 쓰는 tr_id와 반드시 일치시켜야 함(다르면 403/404 가능)

        append_raw_row(
            raw_csv_path="data/raw.csv",
            base_url=getattr(config, "api_url", getattr(config, "base_url", "")),
            endpoint=endpoint,
            tr_id=tr_id,
            params=params,
            status_code=None,
            response_obj=news_data,
        )


        
        # 정규화된 데이터 저장
        records = None

        if isinstance(news_data, list):
            records = news_data

        elif isinstance(news_data, dict):
            for key in ("output", "output1", "output2", "data", "items", "list"):
                if key in news_data and news_data[key] is not None:
                    records = news_data[key]
                    break

            if isinstance(records, dict):
                for key in ("output", "list", "items", "data"):
                    if key in records:
                        records = records[key]
                        break

        if isinstance(records, dict):
            records = [records]

        if isinstance(records, list) and len(records) > 0:
            saver.save_normalized_data(records)
        else:
            print("⚠️ 정규화 대상 레코드를 찾지 못했습니다. raw_json 구조를 확인하세요.")

    
    else:
        print("❌ 뉴스 데이터 조회 실패.")
    
    print("🎉 데이터 수집 완료")


if __name__ == "__main__":
    main()