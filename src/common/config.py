import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()


class KISConfig:
    """KIS API 설정 클래스"""

    def __init__(self):
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.is_paper = os.getenv("KIS_IS_PAPER", "true").lower() == "true"

        # 실전/모의투자 서버 URL 설정
        if self.is_paper:
            # 모의투자(VTS)
            self.api_url = "https://openapivts.koreainvestment.com:29443"
        else:
            # 실서버
            self.api_url = "https://openapi.koreainvestment.com:9443"

    def validate(self) -> bool:
        """필수 설정값 검증"""
        if not self.app_key or not self.app_secret:
            print("❌ KIS_APP_KEY 또는 KIS_APP_SECRET가 설정되지 않았습니다")
            return False

        print("✅ 환경설정 확인 완료")
        print(f"   - 모의투자: {self.is_paper}")
        print(f"   - API URL: {self.api_url}")
        return True


class KISAuth:
    """KIS 인증 클래스 (토큰 파일 캐시 재사용)"""

    def __init__(self, config: KISConfig):
        self.config = config
        self.access_token: Optional[str] = None

    def get_access_token(self) -> Optional[str]:
        """
        접근 토큰 가져오기
        - src/auth/token_manager.py 에서 파일 캐시(.secrets/kis_token.json)를 읽고
          만료 시에만 재발급하도록 구현되어 있어야 함
        """
        try:
            # token_manager.py는 네가 이미 만든 파일 기준
            from .auth.token_manager import get_access_token as get_cached_token

            self.access_token = get_cached_token(
                base_url=self.config.api_url,
                appkey=self.config.app_key,
                appsecret=self.config.app_secret,
            )
            return self.access_token
        except Exception as e:
            print(f"❌ 토큰 로드/재발급 중 에러: {e}")
            return None

    def get_headers(self, tr_id: str) -> Optional[Dict[str, Any]]:
        """인증 헤더 생성"""
        if not self.access_token:
            self.get_access_token()

        if not self.access_token:
            print("❌ 접근 토큰 발급 실패로 헤더 생성 불가")
            return None

        return {
            # 헤더 대소문자는 보통 무관하지만, 문서/포털 표기를 따라 lowercase로 통일
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id,
        }
