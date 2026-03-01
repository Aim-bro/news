import json
import time
from pathlib import Path

import requests


TOKEN_PATH = Path(".secrets/kis_token.json")
TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)


def _now() -> int:
    return int(time.time())


def load_cached_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached_token(token_json: dict) -> None:
    tmp = TOKEN_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(token_json, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(TOKEN_PATH)  # atomic on most OS


def get_access_token(
    *,
    base_url: str,
    appkey: str,
    appsecret: str,
    safety_margin_sec: int = 300,  # 5분 여유
) -> str:
    cached = load_cached_token()
    if cached:
        expires_at = int(cached.get("expires_at", 0))
        if expires_at - _now() > safety_margin_sec:
            return str(cached["access_token"])

    # 만료/없음 → 재발급
    url = f"{base_url}/oauth2/tokenP"
    headers = {"content-type": "application/json; charset=utf-8"}
    body = {
        "grant_type": "client_credentials",
        "appkey": appkey,
        "appsecret": appsecret,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    token = resp.json()

    # 문서/응답에 expires_in(초) 제공되는 케이스가 많음(보통 86400)
    issued_at = _now()
    expires_in = int(token.get("expires_in", 86400))
    token["issued_at"] = issued_at
    token["expires_at"] = issued_at + expires_in

    save_cached_token(token)

    return str(token["access_token"])
