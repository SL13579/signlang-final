# app/sequence_buffer.py
from collections import deque

# 최대 자모 수 설정 (예: 여의도 → 3~4자, 버퍼는 넉넉히 10자까지)
MAX_BUFFER_SIZE = 10

# 자모 인식 버퍼
sequence_buffer = deque(maxlen=MAX_BUFFER_SIZE)

def append_char(char: str):
    """새 자모 추가"""
    if len(char) == 1:
        sequence_buffer.append(char)

def get_combined_text() -> str:
    """버퍼의 자모들을 문자열로 결합"""
    return ''.join(sequence_buffer)

def clear_buffer():
    sequence_buffer.clear()
