import os
import json

# 경로 설정
splits = ["train", "val"]
base_image_dir = "datasets/images"
base_label_dir = "datasets/labels"
os.makedirs(base_label_dir, exist_ok=True)

# 전체 이미지 클래스 추출
all_classes = set()
for split in splits:
    split_path = os.path.join(base_image_dir, split)
    for fname in os.listdir(split_path):
        if fname.endswith((".png", ".jpg")):
            name = os.path.splitext(fname)[0]
            class_name = name if name in ["hello_1", "hello_2", "thankyou_1", "thankyou_2"] else name.rsplit("_", 1)[0]
            all_classes.add(class_name)

# 정렬 및 인덱싱
class_names = sorted(all_classes)
class_to_id = {name: idx for idx, name in enumerate(class_names)}

# label_map.json 저장
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(class_to_id, f, ensure_ascii=False, indent=2)

# 각 이미지에 대해 .txt 생성
for split in splits:
    image_dir = os.path.join(base_image_dir, split)
    label_dir = os.path.join(base_label_dir, split)
    os.makedirs(label_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if not fname.endswith((".png", ".jpg")):
            continue

        name = os.path.splitext(fname)[0]
        class_name = name if name in ["hello_1", "hello_2", "thankyou_1", "thankyou_2"] else name.rsplit("_", 1)[0]
        class_id = class_to_id.get(class_name, -1)
        if class_id == -1:
            print(f"⚠️ 클래스 인식 실패: {fname}")
            continue

        label_path = os.path.join(label_dir, name + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print(f"✅ .txt 라벨 생성 완료 ({len(class_to_id)}개 클래스)")
