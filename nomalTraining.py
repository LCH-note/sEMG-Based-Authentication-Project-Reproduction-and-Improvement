import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
import dataLoad as dataLoad  # dataLoad.py에서 load_data() 함수를 가져옵니다.
from pathlib import Path

# 1. 데이터셋 로드
PATH = "data/semg_dataset.npz"  # 데이터셋 경로
Xtr, ytr, Xte, yte = dataLoad.load_data(PATH)

# GPU/CPU 장치 설정
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {dev}")

# DataLoader 생성 (batch_size=16)
# Xtr, ytr, Xte, yte 가 메모리에 준비되어 있어야 합니다.
train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
test_dataset  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))

BATCH_SIZE = 16
CHECKPOINT_PATH = Path("artifacts/checkpoints/densenet161/standard/best_densenet161.pth")
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. DenseNet161 모델 빌드 (Scratch 학습 + Dropout 0.2 적용)
model = densenet161(weights=None)  # 사전학습 가중치 배제
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),             # 논문 기준 과적합 방지 드롭아웃
    nn.Linear(2208, 5)             # 5명 분류기
)
model = model.to(dev)

# 4. 옵티마이저 및 손실함수 설정
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 5. 45 Epoch 학습 및 실시간 검증 루프
EPOCHS = 45
best_acc = 0.0

for epoch in range(EPOCHS):
    # --- [1] Train Step ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)

        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()

        train_loss += loss.item() * len(yb)
        train_correct += (out.argmax(dim=1) == yb).sum().item()
        train_total += len(yb)

    tr_loss = train_loss / train_total
    tr_acc = (train_correct / train_total) * 100.0

    # --- [2] Evaluation Step ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            loss = crit(out, yb)

            val_loss += loss.item() * len(yb)
            val_correct += (out.argmax(dim=1) == yb).sum().item()
            val_total += len(yb)

    te_loss = val_loss / val_total
    te_acc = (val_correct / val_total) * 100.0

    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
          f"Test Loss: {te_loss:.4f} Acc: {te_acc:.2f}%")

print(f"\n최고 테스트 정확도: {best_acc:.2f}% (가중치 파일: {CHECKPOINT_PATH} 저장 완료)")
