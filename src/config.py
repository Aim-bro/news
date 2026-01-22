import os
import json
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class KISConfig:
    """KIS API 설정 클래스"""
    
    def __init__(self):
        self.app_key = os.getenv('KIS_APP_KEY')
        self.app_secret = os.getenv('KIS_APP_SECRET')
        self.is_paper = os.getenv('KIS_IS_PAPER', 'true').lower() == 'true'
        
        # 실전/모의투자 서버 URL 설정
        if self.is_paper:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.api_url = "https://openapi.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.api_url = "https://openapi.koreainvestment.com:9443"
    
    def validate(self) -> bool:
        """필수 설정값 검증"""
        if not self.app_key or not self.app_secret:
            print("❌ KIS_APP_KEY 또는 KIS_APP_SECRET가 설정되지 않았습니다.")
            return False
        
        if not self.app_key.startswith('P') and not self.app_key.startswith('ps'):
            print("❌ APP_KEY는 'P' 또는 'ps'로 시작해야 합니다.")
            return False
        
        print(f"✅ 환경설정 확인 완료")
        print(f"   - 모의투자: {self.is_paper}")
        print(f"   - API URL: {self.api_url}")
        return True


class KISAuth:
    """KIS 인증 클래스"""
    
    def __init__(self, config: KISConfig):
        self.config = config
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> Optional[str]:
        """접근 토큰 발급"""
        url = f"{self.config.api_url}/oauth2/tokenP"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret
        }
        
        try:
            print("🔐 접근 토큰 발급 중...")
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                self.token_expires_at = datetime.now(timezone.utc)
                
                print("✅ 접근 토큰 발급 성공")
                return self.access_token
            else:
                print(f"❌ 토큰 발급 실패: {response.status_code}")
                print(f"   - 응답: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 중 에러 발생: {e}")
            return None
    
    def get_headers(self, tr_id: str = "FHPST01010400") -> Optional[dict]:
        """인증 헤더 생성"""
        if not self.access_token:
            self.get_access_token()
        if not self.access_token:
            print("❌ 접근 토큰 발급 실패로 헤더 생성 불가")
            return None

                
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id
        }