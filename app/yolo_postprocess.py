from collections import deque

# 쌍자음 및 이중모음 병합 맵
merge_map = {
    ("ㄱ", "ㄱ"): "ㄲ",
    ("ㄷ", "ㄷ"): "ㄸ",
    ("ㅂ", "ㅂ"): "ㅃ",
    ("ㅅ", "ㅅ"): "ㅆ",
    ("ㅈ", "ㅈ"): "ㅉ",
    ("ㅗ", "ㅏ"): "ㅘ",
    ("ㅗ", "ㅐ"): "ㅙ",
    ("ㅜ", "ㅓ"): "ㅝ",
    ("ㅜ", "ㅔ"): "ㅞ",
}

# 예측 결과를 담을 버퍼
prediction_buffer = deque(maxlen=5)

def process_prediction(new_pred: str) -> str:
    prediction_buffer.append(new_pred)

    if len(prediction_buffer) >= 2:
        last_two = tuple(prediction_buffer)[-2:]
        merged = merge_map.get(last_two)
        if merged:
            prediction_buffer.clear()
            return merged

    return new_pred
