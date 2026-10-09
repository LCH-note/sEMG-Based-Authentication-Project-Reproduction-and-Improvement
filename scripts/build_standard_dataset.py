import os
import numpy as np, glob
from semg_auth.preprocessing import HOP, WIN, make_windows, minmax, preprocess, to_cwt

# 데이터셋 만들기

files = sorted(glob.glob('data/raw/**/*.csv', recursive=True))

from sklearn.model_selection import train_test_split

# 1) CSV를 읽고 (3000, 2) 모양을 보장하는 함수
def load(f):
    x = np.loadtxt(f, delimiter=',', skiprows=1)
    if x.shape[0] < x.shape[1]:  # (채널, 시간)으로 들어올 경우 (시간, 채널)로 전치
        x = x.T
    return x

# 2) 폴더명에서 피험자 번호를 추출하여 0 ~ 4 정수로 반환하는 함수
def label_of(f):
    sub = os.path.basename(os.path.dirname(f))
    subject_label_int = ord(sub) - ord('A')
    return subject_label_int

# 3) 전체 파일에 대해 정수 라벨 리스트 생성
labels = [label_of(f) for f in files]

# 4) 시행 단위로 먼저 분할 ← 순서가 핵심
tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

# 5) 각 집합 안에서만 윈도우 생성
def build(flist):
  X, y = [], []
  for f in flist:
    sig = preprocess(load(f))
    for w in minmax(make_windows(sig)):
      X.append(to_cwt(w)); y.append(label_of(f))
  return np.stack(X), np.array(y)


if __name__== '__main__':
    Xtr, ytr = build(tr_files)
    Xte, yte = build(te_files)

    # 저장할 대상 폴더 경로 설정 및 폴더 생성 (폴더가 없으면 자동 생성)
    save_dir = f"data/processed/win{WIN}_hop{HOP}"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터셋을 압축 형식(.npz)으로 저장
    save_path = os.path.join(save_dir, "semg_dataset.npz")
    np.savez_compressed(save_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    print(f"데이터셋 저장 완료!\n저장 경로: {save_path}")

    # 정상적으로 저장되었는지 파일 용량 확인
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"파일 크기: {file_size_mb:.2f} MB")
