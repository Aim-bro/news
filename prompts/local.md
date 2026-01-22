# 1. 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수
copy .env.example .env
# .env에 키 입력

# 4. 실행
python -m src.main
