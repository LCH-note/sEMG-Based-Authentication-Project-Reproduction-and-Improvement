"""개선된 5-fold 교차검증, 전체 train 재학습, 독립 test 평가.

핵심 원칙
1. 각 fold의 최고 검증 모델만 저장하고 patience 기반으로 조기 종료한다.
2. 최고 fold 가중치를 전체 데이터에 이어서 학습하지 않는다.
3. CV의 best epoch 중앙값을 이용해 최종 epoch 수를 정하고 새 모델을 학습한다.
4. 독립 test는 최종 모델 학습이 끝난 뒤 한 번만 평가한다.

실행 예시:
    python five_fold_improved.py --model efficientnet_b0
"""

import argparse
import json
import random
from pathlib import Path

import dataLoad
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import (
    densenet161,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
)


MODEL_CHOICES = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}

NUM_CLASSES = 5
N_FOLDS = 5
SEED = 42
DEFAULT_MODEL = "densenet161"  # 사용할 모델 이름을 선택합니다. (densenet161, resnet18, efficientnet_b0, mobilenet_v3_large)
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 12
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DATA_DIR = Path("data/5fold dataset")
SAVE_METRICS = Path("artifacts/metrics")
CLASS_NAMES = [f"subject_{chr(ord('A') + i)}" for i in range(NUM_CLASSES)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help=f"학습할 모델(기본값: {DEFAULT_MODEL})",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    return parser.parse_args()


def validate_args(args):
    if args.max_epochs < 1:
        raise ValueError("max-epochs는 1 이상이어야 합니다.")
    if args.patience < 1:
        raise ValueError("patience는 1 이상이어야 합니다.")
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate는 0보다 커야 합니다.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay는 0 이상이어야 합니다.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(X, y, batch_size, shuffle=False, seed=SEED):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # CWT scale 수와 채널 수만 고정하고 window 길이는 데이터에서 읽는다.
    if X.ndim != 4 or X.shape[1] != 3 or X.shape[2] != 32:
        raise ValueError(f"입력은 (N, 3, 32, window) 형태여야 합니다: {X.shape}")
    if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("샘플 수와 라벨 모양을 확인하세요. 빈 데이터는 사용할 수 없습니다.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("라벨은 정수형이어야 합니다.")
    if np.any((y < 0) | (y >= NUM_CLASSES)):
        raise ValueError(f"라벨은 0~{NUM_CLASSES - 1} 범위여야 합니다.")
    if not np.isfinite(X).all():
        raise ValueError("입력에 NaN 또는 무한대가 있습니다.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def fresh_model(device, model_name):
    model = MODEL_CHOICES[model_name](weights=None)

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
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    return model.to(device)


def make_optimizer(model, learning_rate, weight_decay):
    # AdamW의 weight decay로 train 정확도 100% 부근의 과적합을 완화한다.
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
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

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
    )


def evaluate_with_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    logits_parts = []
    label_parts = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            outputs = model(xb)
            total_loss += criterion(outputs, yb).item() * len(yb)
            logits_parts.append(outputs.cpu().numpy())
            label_parts.append(yb.cpu().numpy())

    logits = np.concatenate(logits_parts)
    labels = np.concatenate(label_parts)
    predictions = logits.argmax(axis=1)
    accuracy = 100.0 * np.mean(predictions == labels)
    return total_loss / len(labels), accuracy, labels, predictions, logits


def prediction_metrics(labels, predictions):
    return {
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def read_file_names(npz_path, key):
    """저장된 원본 CSV 순서를 읽는다. 없는 경우 빈 배열을 반환한다."""
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data.files:
            return np.array([], dtype=str)
        return np.asarray(data[key], dtype=str)


def aggregate_file_predictions(logits, labels, file_names):
    """연속 저장된 동일 개수 window를 CSV 파일 단위 예측으로 합친다.

    현재 데이터셋은 모든 CSV가 같은 길이이므로 파일마다 window 수가 같다.
    향후 파일 길이가 달라지면 데이터 생성 단계에서 window별 group_id를
    저장해야 하며, 이 함수는 안전하게 None을 반환한다.
    """
    if len(file_names) == 0 or len(labels) % len(file_names) != 0:
        return None

    windows_per_file = len(labels) // len(file_names)
    grouped_labels = labels.reshape(len(file_names), windows_per_file)
    if not np.all(grouped_labels == grouped_labels[:, :1]):
        return None

    grouped_logits = logits.reshape(
        len(file_names), windows_per_file, NUM_CLASSES
    ).mean(axis=1)
    file_labels = grouped_labels[:, 0]
    file_predictions = grouped_logits.argmax(axis=1)
    result = prediction_metrics(file_labels, file_predictions)
    result.update(
        {
            "accuracy": float(100.0 * np.mean(file_predictions == file_labels)),
            "file_count": int(len(file_names)),
            "windows_per_file": int(windows_per_file),
            "aggregation": "mean_logits",
        }
    )
    return result


def is_better(accuracy, loss, best_accuracy, best_loss):
    return accuracy > best_accuracy or (
        np.isclose(accuracy, best_accuracy) and loss < best_loss
    )


def train_cross_validation(args, device, criterion, save_cv):
    fold_results = []
    oof_labels = []
    oof_predictions = []
    oof_file_labels = []
    oof_file_predictions = []

    for fold in range(1, N_FOLDS + 1):
        set_seed(SEED + fold)
        Xtr, ytr, Xva, yva = dataLoad.load_5fold_data(fold=fold)
        input_shape = list(Xtr.shape[1:])
        train_loader = make_loader(
            Xtr, ytr, args.batch_size, shuffle=True, seed=SEED + fold
        )
        val_loader = make_loader(Xva, yva, args.batch_size)
        model = fresh_model(device, args.model)
        optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
        )

        best_acc = -1.0
        best_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        fold_path = save_cv / f"best_{args.model}_fold{fold}.pth"
        history = []
        print(f"\n=============== Fold {fold} ===============")

        for epoch in range(1, args.max_epochs + 1):
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            improved = is_better(va_acc, va_loss, best_acc, best_loss)

            if improved:
                best_acc = va_acc
                best_loss = va_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), fold_path)
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "saved": improved,
                }
            )
            scheduler.step()

            if epoch % 5 == 0 or improved or epoch == args.max_epochs:
                print(
                    f"Epoch {epoch:02d}/{args.max_epochs} | "
                    f"Train Loss {tr_loss:.4f}, Acc {tr_acc:.2f}% | "
                    f"Val Loss {va_loss:.4f}, Acc {va_acc:.2f}%"
                )

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: {args.patience} epoch 동안 개선 없음 "
                    f"(best epoch {best_epoch})"
                )
                break

        weights = torch.load(fold_path, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        del weights
        best_eval_loss, best_eval_acc, labels, predictions, logits = (
            evaluate_with_predictions(model, val_loader, criterion, device)
        )
        oof_labels.append(labels)
        oof_predictions.append(predictions)

        val_files = read_file_names(
            DATA_DIR / f"fold{fold}_dataset.npz",
            "val_files",
        )
        file_result = aggregate_file_predictions(logits, labels, val_files)
        if file_result is not None:
            windows_per_file = file_result["windows_per_file"]
            grouped_labels = labels.reshape(len(val_files), windows_per_file)[:, 0]
            grouped_logits = logits.reshape(
                len(val_files), windows_per_file, NUM_CLASSES
            ).mean(axis=1)
            oof_file_labels.append(grouped_labels)
            oof_file_predictions.append(grouped_logits.argmax(axis=1))

        fold_results.append(
            {
                "fold": fold,
                "train_samples": int(len(Xtr)),
                "val_samples": int(len(Xva)),
                "input_shape": input_shape,
                "best_epoch": best_epoch,
                "best_acc": float(best_eval_acc),
                "best_loss": float(best_eval_loss),
                "stopped_epoch": int(history[-1]["epoch"]),
                "last_acc": float(history[-1]["val_acc"]),
                "checkpoint": str(fold_path),
                "history": history,
            }
        )
        print(
            f"Fold {fold} 최고 정확도: {best_eval_acc:.2f}% "
            f"(Epoch {best_epoch})"
        )

        del model, optimizer, scheduler, train_loader, val_loader
        del Xtr, ytr, Xva, yva, logits, labels, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_labels = np.concatenate(oof_labels)
    oof_predictions = np.concatenate(oof_predictions)
    oof_metrics = prediction_metrics(oof_labels, oof_predictions)
    oof_metrics["accuracy"] = float(
        100.0 * np.mean(oof_predictions == oof_labels)
    )

    oof_file_metrics = None
    if len(oof_file_labels) == N_FOLDS:
        labels = np.concatenate(oof_file_labels)
        predictions = np.concatenate(oof_file_predictions)
        oof_file_metrics = prediction_metrics(labels, predictions)
        oof_file_metrics["accuracy"] = float(
            100.0 * np.mean(predictions == labels)
        )
        oof_file_metrics["file_count"] = int(len(labels))
        oof_file_metrics["aggregation"] = "mean_logits"

    return fold_results, oof_metrics, oof_file_metrics


def train_final_model(args, device, criterion, fold_results, save_final):
    set_seed(SEED + 100)
    Xtr, ytr = dataLoad.load_type_data(type="train")
    full_train_samples = len(Xtr)

    median_best_epoch = float(
        np.median([row["best_epoch"] for row in fold_results])
    )
    mean_fold_train_samples = float(
        np.mean([row["train_samples"] for row in fold_results])
    )
    sample_ratio = mean_fold_train_samples / full_train_samples
    final_epochs = max(1, int(round(median_best_epoch * sample_ratio)))

    print(
        "\n========== 전체 train 새 모델 학습 =========="
        f"\nCV best epoch 중앙값: {median_best_epoch:.1f}"
        f"\n샘플 수 보정 비율: {sample_ratio:.3f}"
        f"\n최종 학습 epoch: {final_epochs}"
    )

    full_loader = make_loader(
        Xtr,
        ytr,
        args.batch_size,
        shuffle=True,
        seed=SEED + 100,
    )
    # 선택 fold의 가중치를 재사용하지 않고 새 모델에서 시작한다.
    model = fresh_model(device, args.model)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=final_epochs,
    )
    history = []

    for epoch in range(1, final_epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, full_loader, criterion, device, optimizer
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": tr_loss,
                "train_acc": tr_acc,
            }
        )
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{final_epochs} | "
            f"Train Loss {tr_loss:.4f}, Acc {tr_acc:.2f}%"
        )

    final_path = save_final / f"final_{args.model}.pth"
    torch.save(model.state_dict(), final_path)
    input_shape = list(Xtr.shape[1:])
    del full_loader, Xtr, ytr, optimizer, scheduler
    return model, final_path, final_epochs, history, input_shape


def evaluate_independent_test(args, model, device, criterion):
    Xtest, ytest = dataLoad.load_type_data(type="test")
    test_loader = make_loader(Xtest, ytest, args.batch_size, shuffle=False)
    test_loss, test_acc, labels, predictions, logits = evaluate_with_predictions(
        model,
        test_loader,
        criterion,
        device,
    )
    window_metrics = prediction_metrics(labels, predictions)
    window_metrics.update(
        {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "sample_count": int(len(labels)),
        }
    )

    test_files = read_file_names(DATA_DIR / "test_dataset.npz", "files")
    file_metrics = aggregate_file_predictions(logits, labels, test_files)
    del test_loader, Xtest, ytest, logits, labels, predictions
    return window_metrics, file_metrics


def main():
    args = parse_args()
    validate_args(args)
    set_seed(SEED)

    save_cv = Path(
        f"artifacts/checkpoints/{args.model}/improved/cross_validation"
    )
    save_final = Path(f"artifacts/checkpoints/{args.model}/improved/final")
    save_cv.mkdir(parents=True, exist_ok=True)
    save_final.mkdir(parents=True, exist_ok=True)
    SAVE_METRICS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    fold_results, oof_metrics, oof_file_metrics = train_cross_validation(
        args,
        device,
        criterion,
        save_cv,
    )
    scores = np.array([row["best_acc"] for row in fold_results])
    last_scores = np.array([row["last_acc"] for row in fold_results])
    print(
        f"\n5-fold 최고 검증 정확도: "
        f"{scores.mean():.2f}% ± {scores.std():.2f}%"
    )

    model, final_path, final_epochs, full_history, input_shape = train_final_model(
        args,
        device,
        criterion,
        fold_results,
        save_final,
    )

    # 독립 test는 모든 학습과 설정 결정이 끝난 이후 여기서 한 번만 읽는다.
    test_metrics, test_file_metrics = evaluate_independent_test(
        args,
        model,
        device,
        criterion,
    )

    summary = {
        "model_name": args.model,
        "seed": SEED,
        "input_shape": input_shape,
        "max_cv_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "fold_results": fold_results,
        "cv_best_mean": float(scores.mean()),
        "cv_best_std": float(scores.std()),
        "cv_stopped_last_mean": float(last_scores.mean()),
        "cv_stopped_last_std": float(last_scores.std()),
        "cv_oof_window_metrics": oof_metrics,
        "cv_oof_file_metrics": oof_file_metrics,
        "final_epochs": final_epochs,
        "final_epoch_rule": (
            "round(median_cv_best_epoch * "
            "mean_fold_train_samples / full_train_samples)"
        ),
        "full_train_history": full_history,
        "final_model": str(final_path),
        "final_training": "fresh_model_on_full_train_cv_derived_epochs",
        "independent_test_evaluated": True,
        "test_evaluation_count": 1,
        "test_window_metrics": test_metrics,
        "test_file_metrics": test_file_metrics,
        "test_policy": (
            "test 결과는 보고 전용이며 모델/하이퍼파라미터 선택에 사용하지 않음"
        ),
    }
    metrics_path = SAVE_METRICS / f"results_5fold_improved_{args.model}.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n독립 Test Loss: {test_metrics['loss']:.4f}")
    print(f"독립 Test window 정확도: {test_metrics['accuracy']:.2f}%")
    if test_file_metrics is not None:
        print(f"독립 Test file 정확도: {test_file_metrics['accuracy']:.2f}%")
    print(f"최종 모델: {final_path.resolve()}")
    print(f"결과 기록: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
