# KIS(Open Trading API) 뉴스(또는 대체 데이터) 수집 MVP 프롬프트 (API 테스트 우선)

너는 시니어 파이썬 엔지니어다.
목표는 **한국투자증권 Open Trading API**로 “뉴스 또는 뉴스에 준하는 데이터(공시/리서치/시황 등)”를
**먼저 API 호출로 테스트**하고, 정상 동작이 확인되면 최소 수집 결과를 **CSV로 저장**하는 것이다.

---

## 0. 절대 규칙
- 한국어로만 답해라
- 문서에 없는 endpoint를 절대 상상해서 만들지 마라
- 뉴스 API가 없으면 없다고 말하고, 대체 가능한 데이터(공시/리서치/시황)를 찾아라
- 모든 비밀값(APP_KEY/SECRET 등)은 `.env`에서 읽고 하드코딩 금지
- 지금은 “거창한 파이프라인” 금지. **테스트 가능한 최소 코드**만 만든다
- 코드 실행 위치(최상위 폴더):
  - `C:\Users\Jo\Downloads\vscode\practice\Josh\news`
  - 이 news 폴더 안에서 모든 걸 해결한다

---

## 1. 현재 단계 목표 (MVP)
1) KIS 인증(토큰 발급) 성공
2) “뉴스 또는 대체 데이터” endpoint 1개를 실제로 호출해 응답 받기
3) 응답 원본(raw_json)과 최소 정규화 결과를 각각 CSV로 저장하기

---

## 2. 먼저 해야 할 일: 문서에서 endpoint 찾기
아래 GitHub 문서를 기준으로 **뉴스/대체 데이터 endpoint를 찾아라**

- https://github.com/koreainvestment/open-trading-api

### 반드시 답할 것
- 뉴스 관련 endpoint가 있는가? 없다면 대체 데이터 endpoint는 무엇인가?
- 해당 endpoint의
  - URL 경로
  - Method
  - TR ID
  - 필수 헤더(authorization 포함)
  - 필수 파라미터
  - 응답 예시에서 핵심 필드

⚠️ 이걸 확인하기 전엔 코드부터 쓰지 마라

---

## 3. 출력 형식 (이 순서 고정)
1) 찾은 endpoint 요약(표)
2) MVP 파일 트리
3) 각 파일의 전체 코드
4) PowerShell 실행 명령어
5) 실패 시 체크리스트(401/403/429/토큰/서버구분)

---

## 4. MVP 파일 구조 (최소)
news/
 ├─ .env.example
 ├─ requirements.txt
 ├─ data/
 │   ├─ raw.csv
 │   └─ norm.csv
 └─ src/
     ├─ config.py        # 환경변수 로드/검증
     ├─ kis_auth.py      # 토큰 발급(최소)
     ├─ kis_client.py    # requests 기반 공통 호출(타임아웃/에러처리 최소)
     ├─ fetch.py         # endpoint 1개 호출해서 데이터 받아오기
     ├─ save_csv.py      # raw/norm csv 저장
     └─ main.py          # python -m src.main 로 실행

---

## 5. CSV 저장 요구사항
- raw.csv: 호출 시각 + endpoint + params + 응답 raw_json(문자열) 형태로 1행씩 누적
- norm.csv: 아래 최소 컬럼을 만들어 가능한 것만 채워 저장
  - fetched_at
  - published_at (가능하면)
  - title (가능하면)
  - source (가능하면)
  - category (가능하면)
  - url (가능하면)
  - id (없으면 title+published_at 해시)

---

## 6. 환경변수(.env)
- KIS_APP_KEY=
- KIS_APP_SECRET=
- KIS_IS_PAPER=true/false
- KIS_BASE_URL= (모의/실서버에 맞게)

`.env.example`도 같이 만들어라

---

## 7. 실행 명령 (PowerShell)
- 가상환경 만들고 설치하는 명령 포함
- 실행:
  - `python -m src.main`

---

## 8. 구현 품질(최소만)
- timeout은 반드시 넣어라(예: 10초)
- HTTP status code별로 에러 메시지 친절하게 출력
- 429면 “잠시 후 재시도 필요” 안내(자동 재시도는 지금 단계에선 옵션)

---

## 9. 마지막에 반드시 포함
- “다음 단계 로드맵”을 5줄 이내로 제시:
  - endpoint 확장
  - 증분 수집(watermark)
  - 스키마 확장
  - 저장소(SQLite/parquet)
  - 스케줄링

지금 바로 2번(문서에서 endpoint 찾기)부터 시작해라
