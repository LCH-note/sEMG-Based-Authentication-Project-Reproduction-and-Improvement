import numpy as np, glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, mobilenet_v3_large, resnet18, efficientnet_b0
from sklearn.model_selection import StratifiedKFold
import dataLoad
from dataset import build, load, label_of
from pathlib import Path

EPOCHS = 45
BATCH_SIZE = 16
NUM_CLASSES = 5
N_FOLDS = 5
MODEL_NAME = "mobilenet_v3_large"  # 사용할 모델 이름을 선택합니다. (densenet161, resnet18, efficientnet_b0, mobilenet_v3_large)
CHECKPOINT_DIR = Path(f"artifacts/checkpoints/{MODEL_NAME}/legacy_cv")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


model_choices = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}


def make_model(device, model_name=MODEL_NAME):
    """프로젝트에서 사용하는 각 모델을 생성합니다."""
    model = model_choices.get(model_name)(weights=None)
    
    if model_name == "densenet161":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif model_name in {"efficientnet_b0", "mobilenet_v3_large"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features,
            NUM_CLASSES,
        )
    return model.to(device)

# GPU/CPU 장치 설정
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {dev}")

# 5-fold cross-validation 설정
scores = []

for fold in range(1, N_FOLDS + 1):
    print(f"\n{'='*15} Fold {fold} {'='*15}")

    # 파일 단위 누수 방지 분할 적용
    Xtr, ytr, Xva, yva = dataLoad.load_5fold_data_all(fold=fold)

    train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    val_dataset  = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. DenseNet161 모델 빌드 (Scratch 학습 + Dropout 0.2 적용)
    model = make_model(dev)

    # 4. 옵티마이저 및 손실함수 설정
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # 5. 45 Epoch 학습 및 실시간 검증 루프, 스케쥴러
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc = -1.0
    best_epoch = 0

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

        # --- [2] Validation Step ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                loss = crit(out, yb)

                val_loss += loss.item() * len(yb)
                val_correct += (out.argmax(dim=1) == yb).sum().item()
                val_total += len(yb)

        va_loss = val_loss / val_total
        va_acc = (val_correct / val_total) * 100.0
        # 현재 Fold에서 가장 좋은 validation 정확도 저장
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch + 1

            torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_{MODEL_NAME}_fold{fold}.pth")
        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
                    print(f"  Epoch [{epoch+1:02d}/{EPOCHS}] | Train Acc: {tr_acc:.2f}% | Val Acc: {va_acc:.2f}%")

    scores.append(best_acc)

    print(f"▶ Fold {fold} 최고 Validation 정확도: "f"{best_acc:.2f}% (Epoch {best_epoch})")

    # 3. 5-Fold 전체 종료 후 논문과 동일한 형태의 통계 출력
print(f"=== 5-Fold 최종 결과 ===")
for fold, score in enumerate(scores, start=1):
    print(f"Fold {fold}: {score:.2f}%")
print(f"평균 정확도: {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")
