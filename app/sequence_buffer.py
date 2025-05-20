from collections import deque
import time

# 최대 자모 수 설정
MAX_BUFFER_SIZE = 10

# 자모 인식 버퍼
sequence_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# 중복 자모 판단용 이전 값과 시간
prev_char = None
prev_time = 0

def append_char_if_new(char: str):
    """2초 내 동일 자모는 무시하고 새로운 자모만 추가"""
    global prev_char, prev_time
    current_time = time.time()

    # 동일 자모가 2초 이내 들어온 경우 무시
    if char == prev_char and (current_time - prev_time) < 2.0:
        return

    sequence_buffer.append(char)
    prev_char = char
    prev_time = current_time

def get_combined_text() -> str:
    """버퍼의 자모들을 문자열로 결합"""
    return ''.join(sequence_buffer)

def clear_buffer():
    """버퍼 초기화"""
    sequence_buffer.clear()
