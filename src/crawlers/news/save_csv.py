import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import os
import json
from pathlib import Path

class KISCSVSaver:
    """KIS 데이터 CSV 저장기"""
    
    def __init__(self):
        # 폴더 생성 확인
        os.makedirs('data', exist_ok=True)
        self.raw_csv_path = 'data/raw.csv'
        self.norm_csv_path = 'data/norm.csv'
    


    


    def save_raw_data(self, endpoint: str, params: Dict[str, Any], 
                   response_data: Dict[str, Any]):
        """원본 데이터 저장"""
        # raw.csv이 존재하는지 확인
        file_exists = os.path.exists(self.raw_csv_path)
        
        # 새 데이터 생성
        new_row = {
            'fetched_at': datetime.now().isoformat(),
            'endpoint': endpoint,
            'params': str(params),
            'raw_response': str(response_data)
        }
        
        # 파일이 존재하면 헤더 없이 새 행 추가
        if file_exists:
            df = pd.DataFrame([new_row])
        else:
            # 첫 행일 경우 헤더와 함께 저장
            df = pd.DataFrame([new_row])
        
        # CSV 파일에 추가 (append)
        df.to_csv(self.raw_csv_path, mode='a', header=not file_exists, 
                 index=False, encoding='utf-8')
        
        print(f"💾 원본 데이터 저장 완료: {self.raw_csv_path}")
    
    def save_normalized_data(self, news_items: List[Dict[str, Any]]):
        """정규화된 데이터 저장"""
        if not news_items:
            return
        
        norm_data = []
        
        for item in news_items:
            # ID 생성 (제목+시간 해시)
            title = item.get('disclosure_title', '')
            published_at = item.get('disclosure_time', '')
            id_source = f"{title}{published_at}"
            data_id = hashlib.md5(id_source.encode()).hexdigest()
            
            norm_row = {
                'fetched_at': datetime.now().isoformat(),
                'published_at': published_at,
                'title': title,
                'source': item.get('disclosure_company', ''),
                'category': item.get('disclosure_gubun', ''),
                'url': item.get('disclosure_url', ''),
                'id': data_id
            }
            
            norm_data.append(norm_row)
        
        df = pd.DataFrame(norm_data)
        
        # 파일이 존재하는지 확인
        file_exists = os.path.exists(self.norm_csv_path)
        
        # CSV 파일에 추가 (append)
        df.to_csv(self.norm_csv_path, mode='a', header=not file_exists, 
                 index=False, encoding='utf-8')
        
        print(f"💾 정규화 데이터 저장 완료: {self.norm_csv_path}")
        print(f"   - {len(norm_data)}건 저장 완료")
        
def append_raw_row(
        *,
        raw_csv_path: str | Path,
        base_url: str,
        endpoint: str,
        tr_id: str,
        params: dict,
        status_code: int | None,
        response_obj,
    ) -> None:
        Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)

        row = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "endpoint": endpoint,
            "tr_id": tr_id,
            "params_json": json.dumps(params, ensure_ascii=False),
            "status_code": status_code,
            "raw_json": json.dumps(response_obj, ensure_ascii=False),
        }

        df = pd.DataFrame([row])

        # 파일 없으면 헤더 포함 생성, 있으면 헤더 없이 append
        write_header = not Path(raw_csv_path).exists()
        df.to_csv(raw_csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")