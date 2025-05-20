# train_char_classifier.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from app.char_classifier import HandCharClassifier

# --- 1. 데이터 로드 ---
X = np.load('data/X_landmarks.npy')  # (N, 63)
y = np.load('data/y_labels.npy')     # (N,)

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. 모델 초기화 ---
model = HandCharClassifier(input_size=63, hidden_size=128, num_classes=40)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 3. 학습 루프 ---
for epoch in range(30):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item()}")

# --- 4. 모델 저장 ---
torch.save(model.state_dict(), 'models/char_classifier.pth')
print("✅ 모델 학습 완료 및 저장: models/char_classifier.pth")
