## 📁 프로젝트 구조

signlang_yolo/
├── app/ # FastAPI 백엔드 앱
│ ├── main.py
│ ├── stt.py
│ ├── sign_detector.py
│ ├── sign_language.py
│ └── i18n.py
├── yolov5/ # YOLOv5 GitHub 클론
├── datasets/ # YOLO 학습용 이미지 및 라벨
├── models/ # 훈련된 YOLO 모델 (.pt)
├── static/ # 정적 자원
├── templates/ # HTML 템플릿
├── requirements.txt # 의존성 목록
├── Dockerfile
└── README.md

yaml
복사
편집

---

## ✅ 설치 및 실행

```bash
python -m venv venv
venv\Scripts\activate  # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

웹 애플리케이션 접속: http://localhost:8080

## 📁 모델 및 데이터셋 다운로드

> GitHub 용량 제한으로 인해 아래 링크에서 리소스를 직접 다운로드해주세요.

🔗 [Google Drive 다운로드](https://drive.google.com/drive/folders/16b7rkyuLWFTZd8PwTpspyjfVWuqLbeRU?usp=sharing)
