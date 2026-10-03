"""Evaluate each final model on the independent test set."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import (
    densenet161,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
)

import dataLoad


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAMES = (
    "densenet161",
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
)
MODEL_BUILDERS = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}
CLASS_NAMES = ("A", "B", "C", "D", "E")
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 16
OUTPUT_DIR = Path("artifacts/evaluation")


def build_model(model_name, device):
    model = MODEL_BUILDERS[model_name](weights=None)
    if model_name == "densenet161":
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier.in_features, NUM_CLASSES),
        )
    elif model_name == "resnet18":
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
    else:
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )

    checkpoint = Path(
        f"artifacts/checkpoints/{model_name}/final/final_{model_name}.pth"
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model.to(device).eval(), checkpoint


def save_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"{model_name} confusion matrix",
    )
    threshold = cm.max() / 2
    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    output_path = OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate(model, loader, device):
    predictions, targets = [], []
    with torch.inference_mode():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    cm = confusion_matrix(targets, predictions, labels=range(NUM_CLASSES))
    class_recall = np.divide(
        np.diag(cm), cm.sum(axis=1), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=1) != 0
    )
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    true_index, predicted_index = np.unravel_index(
        np.argmax(off_diagonal), off_diagonal.shape
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, class_recall)
        },
        "best_class": CLASS_NAMES[int(np.argmax(class_recall))],
        "worst_class": CLASS_NAMES[int(np.argmin(class_recall))],
        "largest_confusion": {
            "true": CLASS_NAMES[true_index],
            "predicted": CLASS_NAMES[predicted_index],
            "count": int(off_diagonal[true_index, predicted_index]),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test, y_test = dataLoad.load_type_data(type="test")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_test, dtype=torch.float32),
            torch.as_tensor(y_test, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    results = {
        "device": str(device),
        "test_samples": int(len(y_test)),
        "averaging": "macro",
        "models": {},
    }
    for model_name in MODEL_NAMES:
        print(f"Evaluating {model_name}...")
        model, checkpoint = build_model(model_name, device)
        metrics = evaluate(model, loader, device)
        image_path = save_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]), model_name
        )
        metrics["checkpoint"] = checkpoint.as_posix()
        metrics["confusion_matrix_image"] = image_path.as_posix()
        results["models"][model_name] = metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = OUTPUT_DIR / "model_comparison.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved evaluation results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
