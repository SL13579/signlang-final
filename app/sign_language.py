# app/sign_language.py
import json
import os

# 현재 파일(app/sign_language.py)의 상위 디렉토리 = 프로젝트 루트
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LABEL_PATH = os.path.join(BASE_DIR, "label_map.json")

# label_map.json 로드
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

def label_to_text(index):
    return label_map.get(str(index), "알 수 없음")
