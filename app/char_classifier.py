# app/char_classifier.py
import torch
import os
import numpy as np

USE_CHAR_MODEL = False  # 필요할 때 True로만 바꿔주면 됨!

model = None
    
# MLP 모델 정의
class HandCharClassifier(torch.nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=31):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 조건부 모델 로드
if USE_CHAR_MODEL:
    os.makedirs('models', exist_ok=True)
    model = HandCharClassifier()
    if not os.path.exists('models/char_classifier.pth'):
        torch.save(model.state_dict(), 'models/char_classifier.pth')
    model.load_state_dict(torch.load('models/char_classifier.pth'))
    model.eval()
else:
    print("⚠️ char_classifier 모델 비활성화 상태입니다.")

# 라벨 매핑 (0~30 → 영어 자모 기준)
label_map = {
    0: 'g', 1: 'n', 2: 'd', 3: 'r', 4: 'm',
    5: 'b', 6: 's', 7: 'ng', 8: 'j', 9: 'ch',
    10: 'k', 11: 't', 12: 'p', 13: 'h',
    14: 'a', 15: 'ya', 16: 'eo', 17: 'yeo', 18: 'o',
    19: 'yo', 20: 'u', 21: 'yu', 22: 'eu', 23: 'i',
    24: 'ae', 25: 'e', 26: 'yae', 27: 'ye',
    28: 'oe', 29: 'wi', 30: 'ui'
}

def predict_char(landmarks_vector: np.ndarray) -> str:
    if model is None:
        print("❌ char_classifier 모델이 로드되지 않았습니다.")
        return '?'

    input_tensor = torch.tensor(landmarks_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_index = torch.argmax(output, dim=1).item()
    return label_map.get(pred_index, '?')