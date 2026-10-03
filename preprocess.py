from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt
import pywt
import numpy as np

# 필터링
FS = 1000
# def preprocess(x, fs=FS):
#     # 1) 60 Hz 전원선 잡음 제거
#     bn, an = iirnotch(60, 30, fs)
#     x = filtfilt(bn, an, x, axis=0)
#     # 2) 20~499 Hz 대역만 통과 (4차)
#     b, a = butter(4, [20/(fs/2), 499/(fs/2)], btype='band')
#     return filtfilt(b, a, x, axis=0)
def preprocess(x, fs=FS):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("입력은 (시간, 2채널) 형태여야 합니다.")

    if not np.isfinite(x).all():
        raise ValueError("입력에 NaN 또는 무한대가 있습니다.")

    if fs / 2 <= 499:
        raise ValueError("현재 필터 설정에는 fs > 998 Hz가 필요합니다.")

    # 60 Hz 노치 필터
    bn, an = iirnotch(w0=60, Q=30, fs=fs)
    x = filtfilt(bn, an, x, axis=0)

    # 20~499 Hz 대역통과 필터
    sos = butter(
        N=4,
        Wn=[20, 499],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(sos, x, axis=0)

#슬라이딩 윈도우 분할
# WIN, HOP = 100, 50 # 샘플 단위 (1000Hz -> 100ms, 50ms)
# WIN, HOP = 200, 100 # 샘플 단위 (1000Hz -> 200ms, 100ms)
# WIN, HOP = 300, 150 # 샘플 단위 (1000Hz -> 300ms, 150ms)
WIN, HOP = 500, 250 # 샘플 단위 (1000Hz -> 500ms, 250ms)
def make_windows(x, win=WIN, hop=HOP):
    """x: (시간, 채널) -> (윈도우수, win, 채널)"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i*hop : i*hop+win] for i in range(n)])

# min-max 정규화
def minmax(w, eps=1e-8):
    """w: (윈도우수, 시간, 채널)"""
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)
    return (w - mn) / (mx - mn + eps)

# CWT변환
# 16,32,64개 스케일 
# number = 17
number = 33
# number = 65
SCALES = np.arange(1, number) # 스케일 32개
def to_cwt(one_window, wavelet='morl'):
    """(300, 2) -> (3, 32, 300)"""
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef)) # (32, 300)
    maps.append((maps[0] + maps[1]) / 2) # 3번째 채널
    return np.stack(maps).astype(np.float32)




