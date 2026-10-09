"""저장된 5-fold 데이터 학습 -> 최고 fold 모델 -> 전체 훈련 -> 테스트."""
import json
import random
from pathlib import Path

from semg_auth import data_loading as dataLoad
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, resnet18, efficientnet_b0, mobilenet_v3_large

EPOCHS = 45
FINAL_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
N_FOLDS = 5
NUM_CLASSES = 5  # dataset.label_of()의 A~E -> 0~4
SEED = 42
MODEL_NAME = "mobilenet_v3_large"  # 사용할 모델 이름을 선택합니다. (densenet161, resnet18, efficientnet_b0, mobilenet_v3_large)
RUN_DIR = Path("artifacts/runs/baseline_win500_hop250")
SAVE_5FOLD = RUN_DIR / "checkpoints" / MODEL_NAME / "cross_validation"
SAVE_FINAL = RUN_DIR / "checkpoints" / MODEL_NAME / "final"
SAVE_METRICS = RUN_DIR / "metrics"

model_choices = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(X, y, shuffle=False):
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    # if X.ndim != 4 or X.shape[1:] != (3, 32, 300):
    #     raise ValueError(f"입력은 (N, 3, 32, 300)이어야 합니다: {X.shape}")
    if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("샘플 수와 라벨 모양을 확인하세요. 빈 데이터는 사용할 수 없습니다.")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= NUM_CLASSES)):
        raise ValueError(f"라벨은 0~{NUM_CLASSES - 1}의 정수여야 합니다.")
    if not np.isfinite(X).all():
        raise ValueError("입력에 NaN 또는 무한대가 있습니다.")
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def fresh_model(device, model_name=MODEL_NAME):
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


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total_correct += (outputs.argmax(dim=1) == yb).sum().item()
            total_samples += len(yb)
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def main():
    set_seed(SEED)
    SAVE_5FOLD.mkdir(parents=True, exist_ok=True)
    SAVE_FINAL.mkdir(parents=True, exist_ok=True)
    SAVE_METRICS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    print(f"Using device: {device}")
    results = []

    # 1. 저장된 분할 사용: 각 fold는 독립된 새 모델로 학습합니다.
    for fold in range(1, N_FOLDS + 1):
        set_seed(SEED + fold)
        Xtr, ytr, Xva, yva = dataLoad.load_5fold_data(fold=fold)
        train_loader = make_loader(Xtr, ytr, shuffle=True)
        val_loader = make_loader(Xva, yva)
        model = fresh_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
        fold_path = SAVE_5FOLD / f"best_{MODEL_NAME}_fold{fold}.pth"
        print(f"\n=============== Fold {fold} ===============")
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            # 정확도 동률이면 해당 epoch의 검증 loss가 낮은 모델을 선택합니다.
            if va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss):
                best_acc, best_loss, best_epoch = va_acc, va_loss, epoch
                torch.save(model.state_dict(), fold_path)
            scheduler.step()
            if epoch % 5 == 0 or epoch == EPOCHS:
                print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss {tr_loss:.4f}, Acc {tr_acc:.2f}% | Val Loss {va_loss:.4f}, Acc {va_acc:.2f}%")
        results.append(dict(fold=fold, best_epoch=best_epoch, best_acc=best_acc,
                            best_loss=best_loss, last_acc=va_acc, path=str(fold_path)))
        print(f"Fold {fold} 최고 정확도: {best_acc:.2f}% (Epoch {best_epoch})")
        del model, optimizer, scheduler, train_loader, val_loader, Xtr, ytr, Xva, yva
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2. 5개 fold의 최고 저장 모델 중 하나를 선택합니다. 테스트는 사용하지 않습니다.
    selected = max(results, key=lambda row: (row["best_acc"], -row["best_loss"]))
    best_path = SAVE_5FOLD / f"best_{MODEL_NAME}_5fold.pth"
    model = fresh_model(device, model_name=MODEL_NAME)
    weights = torch.load(selected["path"], map_location=device, weights_only=True)
    model.load_state_dict(weights)
    del weights
    torch.save(model.state_dict(), best_path)
    scores = np.array([row["best_acc"] for row in results])
    last_scores = np.array([row["last_acc"] for row in results])
    print(f"\n5-fold 최고 검증 정확도: {scores.mean():.2f}% ± {scores.std():.2f}%")
    print(f"45 epoch 검증 정확도: {last_scores.mean():.2f}% ± {last_scores.std():.2f}%")
    print(f"선택 모델: Fold {selected['fold']}, Epoch {selected['best_epoch']}")

    # 3. 선택된 가중치에서 전체 train으로 추가 학습합니다.
    # 이 과정에서는 독립적인 test 데이터를 불러오거나 평가하지 않습니다.
    set_seed(SEED + 100)
    Xtr, ytr = dataLoad.load_type_data(type="train")
    full_loader = make_loader(Xtr, ytr, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    final_path = SAVE_FINAL / f"final_{MODEL_NAME}.pth"
    full_history = []
    print("\n========== 전체 train 추가 학습 ==========")
    for epoch in range(1, FINAL_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, full_loader, criterion, device, optimizer)
        full_history.append(
            dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc)
        )
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{FINAL_EPOCHS} | "
            f"Train Loss {tr_loss:.4f}, Acc {tr_acc:.2f}%"
        )
    torch.save(model.state_dict(), final_path)
    del full_loader, Xtr, ytr, optimizer, scheduler

    # 4. 전체 train 추가 학습이 끝난 모델을 독립 test 데이터로 한 번만 평가합니다.
    weights = torch.load(final_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    del weights
    Xtest, ytest = dataLoad.load_type_data(type="test")
    test_loader = make_loader(Xtest, ytest, shuffle=False)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    del test_loader, Xtest, ytest

    summary = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   model_name=MODEL_NAME, seed=SEED, fold_results=results,
                   selected_fold=selected["fold"],
                   selected_epoch=selected["best_epoch"], cv_best_mean=float(scores.mean()),
                   cv_best_std=float(scores.std()), cv_last_mean=float(last_scores.mean()),
                   cv_last_std=float(last_scores.std()),
                   validation_source="cross_validation_train_split_only",
                   independent_test_evaluated=True,
                   test_evaluation_count=1,
                   test_loss=test_loss,
                   test_acc=test_acc,
                   full_train_history=full_history,
                   best_cv_model=str(best_path), final_model=str(final_path),
                   final_training="selected_fold_weights_plus_full_train_fixed_epochs")
    metrics_path = SAVE_METRICS / f"results_5fold_{MODEL_NAME}.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n전체 train 추가 학습 완료: {FINAL_EPOCHS} Epoch")
    print(f"독립 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    print(f"교차검증 모델: {best_path.resolve()}")
    print(f"최종 모델: {final_path.resolve()}")
    print(f"결과 기록: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
