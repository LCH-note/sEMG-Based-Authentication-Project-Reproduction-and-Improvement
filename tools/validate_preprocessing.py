import sys
from pathlib import Path

# 현재 파일 기준으로 상위 폴더(부모 디렉터리) 경로를 sys.path에 등록
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # 한 단계 위 상위 폴더
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np, glob
from semg_auth.preprocessing import make_windows, minmax, preprocess, to_cwt

files = sorted(glob.glob('data/raw/**/*.csv', recursive=True))
x = np.loadtxt(files[0], delimiter=',', skiprows=1)

# 필터 검증
xf = preprocess(x)
print('전:', x.std(), ' 후:', xf.std())

w = make_windows(xf)
print('윈도우 배열 모양:', w.shape) # (19, 300, 2)

wn = minmax(w)
print('값 범위:', wn.min(), '~', wn.max())

t = to_cwt(wn[0])
print('입력 텐서 모양:', t.shape) # (3, 32, 300)
