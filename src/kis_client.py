import requests
from typing import Optional, Dict, Any

from .config import KISAuth


class KISClient:
    """KIS API 공통 클라이언트"""

    def __init__(self, auth: KISAuth):
        self.auth = auth

    def _handle_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """API 응답 처리"""
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                print("❌ JSON 파싱 실패")
                print(f"   - 응답: {response.text}")
                return None

        if response.status_code == 401:
            print("❌ 인증 실패 (401): 토큰이 만료되었거나 잘못되었습니다")
        elif response.status_code == 403:
            print("❌ 권한 없음 (403): 해당 API에 접근 권한이 없습니다")
        elif response.status_code == 429:
            print("⚠️ 요청 한도 초과 (429): 잠시 후 재시도 필요합니다")
        else:
            print(f"❌ API 호출 실패: {response.status_code}")
            print(f"   - 응답: {response.text}")

        return None

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        tr_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """GET 요청 (401이면 토큰 갱신 후 1회 재시도)"""
        try:
            final_headers = headers
            if final_headers is None:
                if not tr_id:
                    raise ValueError("headers 미지정 시 tr_id는 필수입니다")
                final_headers = self.auth.get_headers(tr_id)
                if final_headers is None:
                    return None

            response = requests.get(url, params=params, headers=final_headers, timeout=30)

            if response.status_code == 401:
                # 토큰 만료 가능성 → 토큰 비우고 재발급 유도 후 1회 재시도
                self.auth.access_token = None

                if headers is None:
                    final_headers = self.auth.get_headers(tr_id)  # type: ignore[arg-type]
                    if final_headers is None:
                        return None

                response = requests.get(url, params=params, headers=final_headers, timeout=30)

            return self._handle_response(response)

        except requests.exceptions.RequestException as e:
            print(f"❌ GET 요청 중 에러: {e}")
            return None
        except Exception as e:
            print(f"❌ GET 처리 중 에러: {e}")
            return None

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        tr_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST 요청 (401이면 토큰 갱신 후 1회 재시도)"""
        try:
            final_headers = headers
            if final_headers is None:
                if not tr_id:
                    raise ValueError("headers 미지정 시 tr_id는 필수입니다")
                final_headers = self.auth.get_headers(tr_id)
                if final_headers is None:
                    return None

            response = requests.post(url, json=data, headers=final_headers, timeout=30)

            if response.status_code == 401:
                self.auth.access_token = None

                if headers is None:
                    final_headers = self.auth.get_headers(tr_id)  # type: ignore[arg-type]
                    if final_headers is None:
                        return None

                response = requests.post(url, json=data, headers=final_headers, timeout=30)

            return self._handle_response(response)

        except requests.exceptions.RequestException as e:
            print(f"❌ POST 요청 중 에러: {e}")
            return None
        except Exception as e:
            print(f"❌ POST 처리 중 에러: {e}")
            return None
