import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from .config import KISConfig, KISAuth


class KISNewsFetcher:
    """KIS 뉴스/공시 데이터 수집기"""
    
    def __init__(self, config: KISConfig):
        self.config = config
        self.auth = KISAuth(config)
        from .kis_client import KISClient
        self.client = KISClient(self.auth)
        self.base_url = f"{config.api_url}/uapi/domestic-stock/v1/quotations/inquire-price"
    
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