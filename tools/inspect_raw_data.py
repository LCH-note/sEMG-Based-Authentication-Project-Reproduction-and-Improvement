import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np, glob, os
from pathlib import Path

# 데이터 확인

files = sorted(glob.glob('data/raw/**/*.csv', recursive=True))

print('파일 개수:', len(files))
print('예시 경로:', files[0])

x = np.loadtxt(files[0], delimiter=',', skiprows=1) # 형식에 맞게 변경
print('배열 모양:', x.shape) # 예: (3000, 2)
print('값 범위:', x.min(), '~', x.max())
# 피험자별 파일 수 세기
from collections import Counter
subs = [os.path.basename(os.path.dirname(f)) for f in files]
print(Counter(subs))


if x.shape[0] < x.shape[1]: x = x.T # (시간, 채널) 로 통일
t = np.arange(len(x)) / 1000.0 # 1000 Hz -> 초 단위
fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
for ch in range(2):
    ax[ch].plot(t, x[:, ch], lw=0.6)
    ax[ch].set_ylabel(f'ch{ch+1}')
for a in ax:
    a.axvline(1.0, color='r', ls='--')
    a.axvline(2.0, color='r', ls='--')
    ax[1].set_xlabel('time (s)')
output_path = Path('artifacts/analysis/signal_example.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(output_path, dpi=120)


# f0, P0 = welch(x[:, 0], fs=FS, nperseg=512)
# f1, P1 = welch(xf[:, 0], fs=FS, nperseg=512)
# plt.figure(figsize=(9, 3))
# plt.semilogy(f0, P0, label='before', lw=0.8)
# plt.semilogy(f1, P1, label='after', lw=0.8)
# plt.axvline(60, color='r', ls='--')
# plt.xlim(0, 300); plt.legend()
# plt.xlabel('Hz'); plt.ylabel('power')
# plt.tight_layout(); plt.savefig('filter_check.png', dpi=120)
