import torch, numpy as np
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, mobilenet_v3_large, resnet18, efficientnet_b0
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)
import dataLoad

NUM_CLASSES = 5
MODEL_NAME = "densenet161"  # 사용할 모델 이름을 선택합니다. (densenet161, resnet18, efficientnet_b0, mobilenet_v3_large)
model_choices = {
    "densenet161": densenet161,
    "resnet18": resnet18,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_large": mobilenet_v3_large,
}

Xte, yte = dataLoad.load_type_data(type="test")

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'test_loader' not in locals():
    test_dataset = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def load_model(device, model_name=MODEL_NAME):
    """프로젝트에서 사용하는 각 모델을 생성합니다."""
    model = model_choices.get(model_name)(weights=None)
    
    if model_name == "densenet161":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, NUM_CLASSES),
        )
        model.load_state_dict(torch.load(f'artifacts/checkpoints/{MODEL_NAME}/final/final_{MODEL_NAME}.pth',map_location=dev,))

    elif model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, NUM_CLASSES),
        )
        model.load_state_dict(torch.load(f'artifacts/checkpoints/{MODEL_NAME}/cross_validation/best_{MODEL_NAME}_5fold.pth',map_location=dev,))

    elif model_name in {"efficientnet_b0", "mobilenet_v3_large"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features,
            NUM_CLASSES,
        )
        model.load_state_dict(torch.load(
            f'artifacts/checkpoints/{MODEL_NAME}/cross_validation/best_{MODEL_NAME}_5fold.pth',
            map_location=dev,
        ))

    return model.to(device)


if 'model' not in locals():
    model = load_model(dev, MODEL_NAME)
    # model = densenet161(weights=None)
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2),
    #     nn.Linear(2208, 5)
    # )
    # model.load_state_dict(torch.load(
    #     'artifacts/checkpoints/resnet18/legacy_cv/best_resnet18_fold4.pth',
    #     map_location=dev,
    # ))
    # model = model.to(dev)

model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(dev))
        preds += out.argmax(1).cpu().tolist()
        trues += yb.tolist()

acc = accuracy_score(trues, preds)
f1 = f1_score(trues, preds, average='macro')

print(f'accuracy {acc:.4f} macro F1 {f1:.4f}')
print(classification_report(trues, preds, digits=3))

cm = confusion_matrix(trues, preds)
print(cm)
