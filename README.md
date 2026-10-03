# sEMG-Based Authentication Project: Reproduction and Improvement

손잡이를 회전할 때 측정한 **2채널 손바닥 표면 근전도(sEMG)** 신호로 사용자 5명(`A`~`E`)을 인증하는 프로젝트입니다. 원시 시계열을 필터링하고 CWT(Continuous Wavelet Transform) 이미지로 변환한 뒤, 네 가지 CNN 모델을 동일한 조건에서 학습·비교했습니다.

이 프로젝트는 원 연구의 DenseNet161 기반 분류를 재현하고, ResNet18·EfficientNet-B0·MobileNetV3-Large까지 비교하여 정확도뿐 아니라 모델 크기와 실행 시간의 균형을 분석합니다.

## 1. 데이터 및 전처리

### 데이터 구성

- 피험자: 5명 (`A`, `B`, `C`, `D`, `E`)
- 시행 횟수: 피험자당 50회, 총 250개 CSV 파일
- 채널: 2개 (`APB`, `ADM`)
- 샘플링 주파수: 1,000 Hz
- 1회 시행: 잡기 1초 → 회전 1초 → 정지 1초, 총 3초
- 분할: 계층화된 파일 단위 80:20 분할 (`random_state=42`)
  - 학습용: 200개 시행, 5,800개 윈도우
  - 독립 테스트용: 50개 시행, 1,450개 윈도우

윈도우를 먼저 만든 뒤 나누지 않고 **CSV 시행 단위로 먼저 분할**하여, 같은 시행에서 나온 구간이 학습 세트와 테스트 세트에 동시에 들어가는 데이터 누수를 방지했습니다. 학습용 200개 시행 안에서 다시 Stratified 5-Fold 교차검증을 수행합니다.

### 전처리 순서

1. 60 Hz 노치 필터(`Q=30`)로 전원선 잡음 제거
2. 4차 Butterworth 20~499 Hz 대역통과 필터
3. 200 samples(200 ms) 윈도우, 100 samples(100 ms) 간격으로 분할
4. 윈도우별 Min-Max 정규화
5. Morlet wavelet과 32개 scale을 이용한 CWT 변환
6. 채널 1, 채널 2, 두 채널 CWT의 평균을 쌓아 `(3, 32, 200)` 입력 생성

전처리된 한 샘플의 형태는 다음과 같습니다.

```text
2-channel signal (200, 2)
        ↓ CWT
channel 1 map (32, 200)
channel 2 map (32, 200)
average map   (32, 200)
        ↓ stack
CNN input (3, 32, 200)
```

## 2. 사용 모델 및 학습 방법

다음 네 모델은 ImageNet 사전학습 가중치를 사용하지 않고(`weights=None`) 처음부터 학습했습니다. 각 모델의 마지막 분류층을 5개 클래스 출력으로 변경했으며, DenseNet161과 ResNet18에는 분류층 앞에 Dropout 0.2를 적용했습니다.

- DenseNet161
- ResNet18
- EfficientNet-B0
- MobileNetV3-Large

공통 학습 설정은 다음과 같습니다.

| 항목 | 설정 |
|---|---:|
| Loss | Cross Entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 16 |
| Epochs | 45 |
| Scheduler | CosineAnnealingLR |
| Folds | Stratified 5-Fold |
| Random seed | 42 |

각 Fold는 새 모델로 독립 학습합니다. 검증 정확도가 가장 높은 체크포인트를 저장하고, 정확도가 같으면 검증 손실이 더 낮은 체크포인트를 선택합니다. 모델 비교용 Accuracy·Precision·Recall·F1-score와 혼동행렬은 저장된 5-Fold 대표 체크포인트를 **학습 및 모델 선택에 사용하지 않은 1,450개 독립 테스트 윈도우**에서 계산했습니다.

## 3. 실행 방법

### 환경 설치

Python 3.13 및 CUDA 지원 GPU 환경에서 실행했습니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 데이터 배치

원본 CSV 파일을 다음 구조로 배치합니다.

```text
data/raw/
├── A/
├── B/
├── C/
├── D/
└── E/
```

각 폴더에는 해당 피험자의 CSV 파일 50개가 있어야 합니다. 원 데이터의 자세한 설명과 라이선스는 [`data/README.md`](data/README.md)와 [`data/LICENSE`](data/LICENSE)를 참고하세요.

### 데이터셋 생성

```bash
python five_fold_data_mk.py
```

위 명령은 독립 학습/테스트 세트와 5개 교차검증 Fold를 `data/5fold dataset/`에 생성합니다.

### 모델 학습

`five_fold_new.py`의 `MODEL_NAME`을 아래 값 중 하나로 설정한 뒤 실행합니다.

```python
MODEL_NAME = "densenet161"
# densenet161, resnet18, efficientnet_b0, mobilenet_v3_large
```

```bash
python five_fold_new.py
```

### 전체 모델 평가 및 혼동행렬 생성

```bash
python evaluation.py
```

평가 결과는 `artifacts/evaluation/model_comparison.json`에, 혼동행렬은 모델별 PNG 파일로 저장됩니다.

## 4. 코드 설명

| 파일 | 설명 |
|---|---|
| `preprocess.py` | 노치/대역통과 필터, 슬라이딩 윈도우, 정규화, CWT 변환 |
| `dataset.py` | 원본 CSV 로드, 사용자 라벨 생성, 파일 단위 train/test 분할 |
| `five_fold_data_mk.py` | 독립 테스트 세트 및 누수 방지 5-Fold 데이터셋 생성 |
| `dataLoad.py` | 일반·Fold·독립 테스트 NPZ 데이터 로드 |
| `five_fold_new.py` | 네 CNN 중 선택한 모델의 5-Fold 학습, 체크포인트 저장, 최종 학습 |
| `evaluation.py` | 네 모델의 독립 테스트 성능 계산 및 혼동행렬 생성 |
| `check.py` | 파라미터 수, 45 Epoch 학습 시간, 샘플당 추론 시간 측정 |
| `testSet.py` | 지정한 단일 체크포인트의 분류 성능을 콘솔에서 확인 |
| `nomalTraining.py` | DenseNet161 기본 train/test 학습 실험 |

## 5. 모델 성능 비교

### 독립 테스트 세트 성능

Precision, Recall, F1-score는 다중 클래스 간 비중을 동일하게 반영한 **Macro average**입니다.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **DenseNet161** | **85.52%** | **85.79%** | **85.52%** | **85.42%** |
| EfficientNet-B0 | 83.59% | 83.84% | 83.59% | 83.27% |
| ResNet18 | 83.24% | 83.23% | 83.24% | 83.14% |
| MobileNetV3-Large | 80.83% | 81.05% | 80.83% | 80.81% |

DenseNet161이 모든 독립 테스트 지표에서 가장 높은 값을 기록했습니다. Dense connection을 통해 여러 수준의 CWT 특징을 재사용하는 구조가 피험자별 세부 시간-주파수 패턴을 구분하는 데 유리했던 것으로 해석할 수 있습니다. 다만 가장 큰 모델인 만큼 학습 및 추론 비용도 가장 높습니다.

### 5-Fold 검증 결과

| Model | 최고 Fold 정확도 | Fold별 최고 정확도 평균 | 45 Epoch 정확도 평균 |
|---|---:|---:|---:|
| **DenseNet161** | **92.24%** | **90.47% ± 1.80%** | **89.79% ± 1.89%** |
| ResNet18 | 91.18% | 89.61% ± 1.60% | 88.92% ± 1.41% |
| EfficientNet-B0 | 91.05% | 89.58% ± 1.47% | 88.97% ± 1.65% |
| MobileNetV3-Large | 89.08% | 88.05% ± 1.15% | 86.87% ± 1.60% |

교차검증 수치는 학습 세트 내부 검증 결과이고, 앞의 성능 표는 완전히 분리한 테스트 세트 결과이므로 직접 같은 의미로 비교하면 안 됩니다. 테스트 정확도가 교차검증 평균보다 낮은 것은 보지 않은 시행에 대한 일반화 난도가 더 높음을 보여줍니다.

### 모델 크기와 실행 비용

아래 시간은 동일한 로컬 환경(NVIDIA GeForce RTX 4070)에서 45 Epoch 학습 및 batch size 1 추론을 측정한 결과입니다.

| Model | 학습 가능 파라미터 | 학습 시간 | 샘플당 추론 시간 |
|---|---:|---:|---:|
| DenseNet161 | 26,483,045 | 907.8초 | 25.731 ms |
| ResNet18 | 11,179,077 | **131.8초** | **3.544 ms** |
| EfficientNet-B0 | **4,013,953** | 335.1초 | 8.839 ms |
| MobileNetV3-Large | 4,208,437 | 238.2초 | 5.353 ms |

정확도를 우선하면 DenseNet161이 가장 적합합니다. 반면 ResNet18은 DenseNet161보다 독립 테스트 정확도가 2.28%p 낮지만 학습 시간은 약 6.9배 짧고 추론은 약 7.3배 빠르므로, 실시간 인증 환경에서는 가장 실용적인 절충안입니다. EfficientNet-B0는 파라미터 수가 가장 적지만 이 실험에서는 ResNet18보다 학습과 추론이 느렸습니다.

## 6. Confusion Matrix 및 분석

혼동행렬은 행이 실제 클래스, 열이 예측 클래스이며 각 클래스의 테스트 샘플 수는 290개입니다.

### DenseNet161

![DenseNet161 confusion matrix](artifacts/evaluation/confusion_matrix_densenet161.png)

- 가장 잘 분류된 클래스: `A` — 281/290, Recall 96.90%
- 가장 많이 오분류된 클래스: `E` — 208/290, Recall 71.72%
- 가장 큰 오분류: `E → B` 34건
- 주요 오분류 유형: `C → D` 22건, `D → C` 23건, `E → B/D` 34/23건

### ResNet18

![ResNet18 confusion matrix](artifacts/evaluation/confusion_matrix_resnet18.png)

- 가장 잘 분류된 클래스: `A` — 284/290, Recall 97.93%
- 가장 많이 오분류된 클래스: `E` — 214/290, Recall 73.79%
- 가장 큰 오분류: `C → D` 32건
- 주요 오분류 유형: `D → C` 28건, `E → B` 31건

### EfficientNet-B0

![EfficientNet-B0 confusion matrix](artifacts/evaluation/confusion_matrix_efficientnet_b0.png)

- 가장 잘 분류된 클래스: `A` — 287/290, Recall 98.97%
- 가장 많이 오분류된 클래스: `E` — 186/290, Recall 64.14%
- 가장 큰 오분류: `E → B` 43건
- 주요 오분류 유형: `C → D` 33건, `E → C` 33건

### MobileNetV3-Large

![MobileNetV3-Large confusion matrix](artifacts/evaluation/confusion_matrix_mobilenet_v3_large.png)

- 가장 잘 분류된 클래스: `A` — 282/290, Recall 97.24%
- 가장 많이 오분류된 클래스: `E` — 203/290, Recall 70.00%
- 가장 큰 오분류: `D → C` 36건
- 주요 오분류 유형: `E → C` 35건, `E → B` 29건

### 공통 오분류 원인 분석

모든 모델이 클래스 `A`는 96.90% 이상 안정적으로 분류했지만 클래스 `E`에서 가장 낮은 Recall을 보였습니다. 또한 `B·C·D·E` 사이의 상호 오분류가 반복되었습니다. 이는 피험자마다 신호 세기에는 차이가 있더라도, 동일한 손잡이 회전 동작에서 발생하는 일부 시간-주파수 패턴이 서로 겹치기 때문으로 볼 수 있습니다. 센서 접촉 위치, 피부 임피던스, 힘의 크기, 회전 속도의 시행별 변화도 클래스 내부 분산을 키울 수 있습니다.

현재 정규화는 윈도우별 최솟값과 최댓값을 기준으로 하므로 사용자별 절대 진폭 차이를 제거합니다. 이 방식은 측정 환경 변화에 강해질 수 있지만, 개인 식별에 유효한 진폭 정보도 함께 줄일 수 있습니다. 향후 RMS, MAV, waveform length 같은 시간영역 특징을 CWT 특징과 결합하거나, 채널별 정규화 및 데이터 증강을 적용하면 특히 클래스 `E`의 분류 성능을 개선할 가능성이 있습니다.

## 7. 최종 결과

- 가장 성능이 좋은 모델: **DenseNet161** — 독립 테스트 Accuracy 85.52%, Macro F1 85.42%
- 가장 성능이 낮은 모델: **MobileNetV3-Large** — 독립 테스트 Accuracy 80.83%, Macro F1 80.81%
- 가장 안정적으로 분류된 클래스: **A**
- 주요 오분류 클래스: **E**, 특히 `E → B`와 `E → C`
- 실시간 적용을 고려한 절충 모델: **ResNet18** — Accuracy 83.24%, 3.544 ms/sample

실험 결과, 네 모델 모두 CWT로 변환한 2채널 sEMG에서 개인별 특징을 학습할 수 있었으며 DenseNet161이 가장 높은 분류 성능을 보였습니다. 그러나 모델 규모가 커지면 정확도가 개선되는 대신 학습 및 추론 비용이 크게 증가했습니다. 따라서 서버나 고성능 GPU 환경에서는 DenseNet161, 지연 시간이 중요한 실제 인증 장치에서는 ResNet18이 더 적합하다고 판단했습니다.

## 8. 생성 파일과 대용량 파일 안내

`*.npz`, `*.pth`, `data/raw/`, 체크포인트와 중간 분석 결과는 용량 또는 데이터 배포 문제로 `.gitignore`에 포함되어 있습니다. GitHub에는 재현 코드, README, 최종 평가 JSON과 혼동행렬 PNG가 올라가며, 대용량 파일은 실행 과정에서 로컬에 생성됩니다.

```text
data/5fold dataset/*.npz
artifacts/checkpoints/**/*.pth
artifacts/metrics/*.json
```

`artifacts/evaluation/`의 최종 평가 JSON과 PNG는 `.gitignore`에서 제외하여 README의 이미지가 GitHub에서도 표시되도록 했습니다.

## 9. 데이터 출처 및 라이선스

데이터셋: *Palm sEMG-based user authentication during doorknob rotation using a convolutional neural network*

- Authors: Yeonjung Shin, Junghun Kim, Sang-Il Choi
- Data license: CC BY 4.0
- IRB: Kyungpook National University Hospital, KNUH 2025

데이터를 재사용할 때는 원 저작자를 표시하고 [`data/LICENSE`](data/LICENSE)의 조건을 따라야 합니다.
