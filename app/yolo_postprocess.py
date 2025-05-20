from collections import deque

# 쌍자음 및 이중모음 병합 맵
merge_map = {
    ('ㄱ', 'ㄱ'): 'ㄲ',
    ('ㄷ', 'ㄷ'): 'ㄸ',
    ('ㅂ', 'ㅂ'): 'ㅃ',
    ('ㅅ', 'ㅅ'): 'ㅆ',
    ('ㅈ', 'ㅈ'): 'ㅉ',
    ('ㅗ', 'ㅏ'): 'ㅘ',
    ('ㅜ', 'ㅓ'): 'ㅝ',
    ('ㅗ', 'ㅐ'): 'ㅙ',
    ('ㅜ', 'ㅔ'): 'ㅞ',
}

# 예측 결과를 담을 버퍼 (짧게 유지)
prediction_buffer = deque(maxlen=3)

def process_prediction(new_pred: str) -> str:
    """
    최근 3개의 예측을 모아 중복 필터링과 병합 처리를 수행
    """
    prediction_buffer.append(new_pred)

    # 버퍼 내 동일 예측 3회 → 확정
    if len(prediction_buffer) == 3 and len(set(prediction_buffer)) == 1:
        confirmed_char = prediction_buffer[0]
        prediction_buffer.clear()
        return confirmed_char

    # 직전 2개 확인 → 병합 처리
    if len(prediction_buffer) >= 2:
        last_two = tuple(prediction_buffer)[-2:]
        merged = merge_map.get(last_two)
        if merged:
            prediction_buffer.clear()
            return merged

    return new_pred
