import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from ...common.config import KISConfig, KISAuth


class KISNewsFetcher:
    """KIS 뉴스/공시 데이터 수집기"""
    
    def __init__(self, config: KISConfig):
        self.config = config
        self.auth = KISAuth(config)
        from ...common.kis_client import KISClient
        self.client = KISClient(self.auth)
        self.base_url = f"{config.api_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        self.news_title_url = f"{config.api_url}/uapi/domestic-stock/v1/quotations/news-title"

    
    def fetch_news(self) -> Optional[List[Dict[str, Any]]]:
        """종합 시황/공시 데이터 조회"""
        print("📰 종합 시황/공시 데이터 조회 중...")
        
        # 파라미터 설정
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": "005930"
        }
        
        # 헤더에 TR_ID 설정
        headers = self.auth.get_headers("FHKST01010100")
        
        try:
            response = self.client.get(self.base_url, params=params, headers=headers)
            
            if response and 'output' in response:
                news_data = response['output']
                print(f"✅ 데이터 조회 성공: {len(news_data)}건")
                return news_data
            else:
                print("❌ 데이터 조회 실패: 응답에 'output' 필드가 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ 데이터 조회 중 에러: {e}")
            return None
        
    def fetch_news_title(
        self,
        *,
        stock_code: str = "005930",   # 종목 지정(전체 조회가 막힐 때 가장 확실)
        date_yyyymmdd: str = "",      # ""이면 현재 기준(서버 스펙)
        hour_hhmmss: str = "000000",  # "" 또는 "000000" 둘 다 가능. 우선 안정적으로 "000000"
        keyword: str = "",            # 제목 검색어
        input_srno: str = "0",        # 페이징 시작(안전하게 0)
        news_ofer_entp_code: str = "",# 제공기관 코드(전체/기본은 보통 "")
        mrkt_cls_code: str = "",      # 시장 분류(기본은 "")
        rank_sort_cls_code: str = "", # 정렬(기본은 "")
    ) -> Optional[List[Dict[str, Any]]]:
        """국내주식 뉴스(제목) 조회(news-title)"""
        print("📰 news-title 조회 중...")

        # news-title는 빈 문자열("") 자체가 의미가 있는 파라미터라
        # requests params에 그대로 실어 보내는 게 핵심
        params = {
            "FID_NEWS_OFER_ENTP_CODE": news_ofer_entp_code,
            "FID_COND_MRKT_CLS_CODE": mrkt_cls_code,
            "FID_INPUT_ISCD": stock_code,
            "FID_TITL_CNTT": keyword,
            "FID_INPUT_DATE_1": date_yyyymmdd,
            "FID_INPUT_HOUR_1": hour_hhmmss,
            "FID_RANK_SORT_CLS_CODE": rank_sort_cls_code,
            "FID_INPUT_SRNO": input_srno,
        }

        # TR_ID: news-title (포털에서 확인된 값 그대로)
        headers = self.auth.get_headers("FHKST01011800")
        if headers is None:
            print("❌ 헤더 생성 실패")
            return None

        try:
            response = self.client.get(self.news_title_url, params=params, headers=headers)

            if not response:
                print("❌ 응답이 비었습니다")
                return None

            # 공통 메타 확인(있으면)
            rt_cd = response.get("rt_cd")
            msg_cd = response.get("msg_cd")
            msg1 = response.get("msg1")
            if rt_cd and rt_cd != "0":
                print(f"❌ 조회 실패: rt_cd={rt_cd}, msg_cd={msg_cd}, msg1={msg1}")
                return None

            output = response.get("output", [])
            if isinstance(output, list):
                print(f"✅ news-title 조회 성공: {len(output)}건 (msg_cd={msg_cd}, msg1={msg1})")
                return output

            # output이 list가 아닌 특이 케이스 방어
            print("❌ 응답에 output(list)가 없습니다")
            return None

        except Exception as e:
            print(f"❌ news-title 조회 중 에러: {e}")
            return None
