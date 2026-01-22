import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from .config import KISConfig, KISAuth


class KISClient:
    """KIS API 공통 클라이언트"""
    
    def __init__(self, auth: KISAuth):
        self.auth = auth
    
    def _handle_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """API 응답 처리"""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("❌ 인증 실패 (401): 토큰이 만료되었거나 잘못되었습니다.")
        elif response.status_code == 403:
            print("❌ 권한 없음 (403): 해당 API에 접근 권한이 없습니다.")
        elif response.status_code == 429:
            print("⚠️  요청 한도 초과 (429): 잠시 후 재시도 필요합니다.")
        else:
            print(f"❌ API 호출 실패: {response.status_code}")
            print(f"   - 응답: {response.text}")
        return None
    
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, 
              headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """GET 요청"""
        try:
            final_headers = headers or self.auth.get_headers()
            response = requests.get(url, params=params, headers=final_headers, timeout=10)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            print(f"❌ GET 요청 중 에러: {e}")
            return None
    
    def post(self, url: str, data: Optional[Dict[str, Any]] = None, 
               headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """POST 요청"""
        try:
            final_headers = headers or self.auth.get_headers()
            response = requests.post(url, json=data, headers=final_headers, timeout=10)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            print(f"❌ POST 요청 중 에러: {e}")
            return None