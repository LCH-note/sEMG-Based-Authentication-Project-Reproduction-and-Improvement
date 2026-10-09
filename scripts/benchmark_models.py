"""모델의 파라미터 수, 학습 시간, 샘플당 추론 시간을 측정합니다."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161, resnet18, efficientnet_b0, mobilenet_v3_large
from semg_auth import data_loading as dataLoad


DATA_PATH = "data/processed/win300_hop150/semg_dataset.npz"
BATCH_SIZE = 16
EPOCHS = 45
LEARNING_RATE = 1e-3
NUM_CLASSES = 5
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 100
MODEL_NAME = "densenet161"  # 사용할 모델 이름을 선택합니다. (densenet161, resnet18, efficientnet_b0, mobilenet_v3_large)

model_choices = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}

def synchronize(device):
    """CUDA 연산이 끝날 때까지 기다려 측정 오차를 방지합니다."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_loaders():
    """프로젝트의 NPZ 데이터를 학습 및 테스트 DataLoader로 변환합니다."""
    Xtr, ytr, Xte, yte = dataLoad.load_data(DATA_PATH)

    train_dataset = TensorDataset(
        torch.as_tensor(Xtr, dtype=torch.float32),
        torch.as_tensor(ytr, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.as_tensor(Xte, dtype=torch.float32),
        torch.as_tensor(yte, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


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


def train_model(model, train_loader, device):
    """프로젝트 설정으로 모델을 학습하고 epoch별 손실을 출력합니다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)

        mean_loss = total_loss / total_samples
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {mean_loss:.4f}")


def measure_inference(model, test_loader, device):
    """배치 크기 1로 한 샘플의 평균 순전파 시간을 ms 단위로 측정합니다."""
    model.eval()
    sample = next(iter(test_loader))[0][:1].to(device)

    with torch.inference_mode():
        # 최초 CUDA 초기화와 커널 선택 비용은 측정에서 제외합니다.
        for _ in range(WARMUP_RUNS):
            model(sample)

        synchronize(device)
        start = time.perf_counter()

        for _ in range(MEASUREMENT_RUNS):
            model(sample)

        synchronize(device)

    elapsed = time.perf_counter() - start
    return elapsed / MEASUREMENT_RUNS * 1000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    train_loader, test_loader = make_loaders()
    model = make_model(device)

    # requires_grad=True인 실제 학습 대상 파라미터만 계산합니다.
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"학습 가능한 파라미터 수: {parameter_count:,}")

    # DataLoader와 모델 생성 시간은 제외하고 45-epoch 학습만 측정합니다.
    synchronize(device)
    start = time.perf_counter()
    train_model(model, train_loader, device)
    synchronize(device)
    training_seconds = time.perf_counter() - start
    print(f"전체 학습 시간: {training_seconds:.1f}초")

    inference_ms = measure_inference(model, test_loader, device)
    print(f"샘플당 평균 추론 시간: {inference_ms:.3f}ms")


if __name__ == "__main__":
    main()
