import hashlib
import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar

EPSILON = 1e-9

def resample(input_data: np.ndarray, ori_sr: int, target_sr: int) -> np.ndarray:
    
    assert ori_sr > 0 and target_sr > 0, "Sampling rate must be positive"

    if abs(ori_sr - target_sr) < EPSILON:
        return input_data
    elif ori_sr > target_sr:
        # Down-sampling
        ratio = int(np.floor(float(ori_sr / target_sr)))
        x, y, z = input_data[::ratio, 0], input_data[::ratio, 1], input_data[::ratio, 2]
        return np.array([x, y, z]).T
    else:
        # Up-sampling
        ratio = int(np.ceil(float(target_sr / ori_sr)))
        x, y, z = [], [], []
        for n in range(input_data.shape[0] - 1):
            x.extend(np.linspace(start=input_data[n, 0], stop=input_data[n+1, 0], num=ratio))
            y.extend(np.linspace(start=input_data[n, 1], stop=input_data[n+1, 1], num=ratio))
            z.extend(np.linspace(start=input_data[n, 2], stop=input_data[n+1, 2], num=ratio))
        return np.array([x, y, z]).T

def sliding_window(input_data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    # Number of windows
    n = int((len(input_data) - window_size) / stride + 1)
    return [input_data[(i * stride):(i * stride + window_size)] for i in range(n)]

def ensure_type(data):
    # ensure data is a numpy array with a supported data type (e.g., float32)
    if isinstance(data, np.ndarray):
        if data.dtype.type is np.object_:
            # change to a supported data type
            data = data.astype(np.float32)
    elif isinstance(data, list):
        data = np.array(data, dtype=np.float32)
    return data

# check MD5 checksum
def check_md5(check_filepath, ref_md5) -> bool:
    md5 = hashlib.md5()
    with check_filepath.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calculated_md5 = md5.hexdigest()

    return calculated_md5 == ref_md5