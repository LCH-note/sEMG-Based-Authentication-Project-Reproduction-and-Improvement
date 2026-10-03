
import os, glob
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from dataset import build, label_of
from pathlib import Path


# 저장할 대상 폴더 경로 설정 및 폴더 생성 (폴더가 없으면 자동 생성)
SAVE_DIR = Path("data/5fold dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def mk_data():

    # 원본파일 불러오기
    files = sorted(glob.glob('data/raw/**/*.csv', recursive=True))

    # 80:20 비율로 train/test 분할
    labels = [label_of(f) for f in files]
    tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

    tr_labels = [label_of(f) for f in tr_files]

    # 5-fold cross-validation 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    Xtr, ytr = build(tr_files)
    Xte, yte = build(te_files)

    # 데이터셋을 압축 형식(.npz)으로 저장
    train_path = SAVE_DIR / "train_dataset.npz"
    test_path = SAVE_DIR / "test_dataset.npz"
    np.savez_compressed(train_path, Xtr=Xtr, ytr=ytr, files=np.asarray(tr_files, dtype=str))
    np.savez_compressed(test_path, Xte=Xte, yte=yte, files=np.asarray(te_files, dtype=str))


    print(f"train/test 데이터셋 저장 완료!\n저장 경로: {SAVE_DIR}")

    # 정상적으로 저장되었는지 파일 용량 확인
    file_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"Train 파일 크기: {file_size_mb:.2f} MB")

    file_size_mb = os.path.getsize(test_path) / (1024 * 1024)
    print(f"Test 파일 크기: {file_size_mb:.2f} MB")


    for fold, (train_idx, val_idx) in enumerate(skf.split(tr_files, tr_labels), start=1):
        print(f"\n{'='*15} Fold {fold} {'='*15}")

        # 파일 단위 누수 방지 분할 적용
        train_files = [tr_files[i] for i in train_idx]
        val_files = [tr_files[i] for i in val_idx]

        # 각 폴드별 데이터 빌드 (위에서 정의한 build 함수 사용)
        Xtr, ytr = build(train_files)
        Xva, yva = build(val_files)

        # 데이터셋을 압축 형식(.npz)으로 저장
        fold_path = SAVE_DIR / f"fold{fold}_dataset.npz"
        np.savez_compressed(fold_path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, train_files=np.asarray(train_files, dtype=str), val_files=np.asarray(val_files, dtype=str))

        print(f"Fold {fold} 데이터셋 저장 완료!\n저장 경로: {SAVE_DIR}")

if __name__ == "__main__":
    mk_data()