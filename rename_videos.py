# rename_videos.py

import os
import re

# 현재 .py 파일이 있는 디렉토리 기준으로 'videos/' 폴더 경로 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')

# 파일명 패턴 예: 0-1_G.mp4 → 0_G_1.mp4
pattern = re.compile(r"^(\d+)-(\d+)_([A-Za-z가-힣]+)\.mp4$")

# 디버그 로그 출력
print(f"🔍 대상 폴더 경로: {VIDEO_DIR}")
print("📁 파일 이름 변경 작업을 시작합니다...\n")

# 변경 수행
for filename in os.listdir(VIDEO_DIR):
    match = pattern.match(filename)
    if match:
        idx, version, label = match.groups()
        new_filename = f"{idx}_{label}_{version}.mp4"
        old_path = os.path.join(VIDEO_DIR, filename)
        new_path = os.path.join(VIDEO_DIR, new_filename)

        # 실제 파일 이름 변경
        os.rename(old_path, new_path)
        print(f"✅ {filename} → {new_filename}")
    else:
        print(f"⚠️ 무시됨 (패턴 불일치): {filename}")

print("\n🎉 모든 파일 이름 변경이 완료되었습니다.")
