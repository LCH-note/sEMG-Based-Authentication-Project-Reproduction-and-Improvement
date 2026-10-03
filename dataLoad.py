import os
import numpy as np

def load_data(file_path="data/semg_dataset.npz"):
    # 스크립트 기준 경로가 필요할 경우 아래 주석 해제
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")

    # with 구문을 통해 안전하게 로드 후 파일 핸들 해제
    with np.load(file_path, allow_pickle=True) as data:
        Xtr, ytr = data['Xtr'], data['ytr']
        Xte, yte = data['Xte'], data['yte']
    print("데이터셋 로드 성공!")
    print(f"Train Shape: {Xtr.shape}, Labels: {ytr.shape}")
    print(f"Test  Shape: {Xte.shape}, Labels: {yte.shape}")

    return Xtr, ytr, Xte, yte


def load_5fold_data(fold = 1, file_path="data/5fold dataset"):

    if fold not in range(1, 6):
        raise ValueError("fold는 1~5 사이의 정수여야 합니다.")
    
    # 스크립트 기준 경로가 필요할 경우 아래 주석 해제
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)

    fold_path = os.path.join(file_path, f"fold{fold}_dataset.npz")

    if not os.path.exists(fold_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(fold_path)}")

    # with 구문을 통해 안전하게 로드 후 파일 핸들 해제
    with np.load(fold_path, allow_pickle=True) as data:
        Xtr, ytr = data['Xtr'], data['ytr']
        Xva, yva = data['Xva'], data['yva']
    print("데이터셋 로드 성공!")
    print(f"Train Shape: {Xtr.shape}, Labels: {ytr.shape}")
    print(f"Validation Shape: {Xva.shape}, Labels: {yva.shape}")

    return Xtr, ytr, Xva, yva

# def load_5fold_data_all(fold = 1, file_path="data/5fold dataset"):

#     if fold not in range(1, 6):
#         raise ValueError("fold는 1~5 사이의 정수여야 합니다.")
    
#     # 스크립트 기준 경로가 필요할 경우 아래 주석 해제
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(base_dir, file_path)

#     fold_path = os.path.join(file_path, f"fold{fold}_dataset_all.npz")

#     if not os.path.exists(fold_path):
#         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(fold_path)}")

#     # with 구문을 통해 안전하게 로드 후 파일 핸들 해제
#     with np.load(fold_path, allow_pickle=True) as data:
#         Xtr, ytr = data['Xtr'], data['ytr']
#         Xva, yva = data['Xva'], data['yva']
#     print("데이터셋 로드 성공!")
#     print(f"Train Shape: {Xtr.shape}, Labels: {ytr.shape}")
#     print(f"Validation Shape: {Xva.shape}, Labels: {yva.shape}")

#     return Xtr, ytr, Xva, yva


def load_type_data(type='train', file_path="data/5fold dataset"):
    if type not in ['train', 'test']:
        raise ValueError("type은 'train' 또는 'test'이어야 합니다.")
    
    # 스크립트 기준 경로가 필요할 경우 아래 주석 해제
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)

    if type == 'train':
        file_path = os.path.join(file_path, "train_dataset.npz")

        if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")

        # with 구문을 통해 안전하게 로드 후 파일 핸들 해제
        with np.load(file_path, allow_pickle=True) as data:
            Xtr, ytr = data['Xtr'], data['ytr']
        print("데이터셋 로드 성공!")
        print(f"Train Shape: {Xtr.shape}, Labels: {ytr.shape}")
    
        return Xtr, ytr

    elif type == 'test':
        file_path = os.path.join(file_path, "test_dataset.npz")

        if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")
        
        # with 구문을 통해 안전하게 로드 후 파일 핸들 해제
        with np.load(file_path, allow_pickle=True) as data:
            Xte, yte = data['Xte'], data['yte']
        print("데이터셋 로드 성공!")
        print(f"Test  Shape: {Xte.shape}, Labels: {yte.shape}")
    
        return Xte, yte


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()