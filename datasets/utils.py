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
        x, y, z = input_data[0][::ratio], input_data[1][::ratio], input_data[2][::ratio]
        return np.array([x, y, z])
    else:
        # Up-sampling
        ratio = int(np.ceil(float(target_sr / ori_sr)))
        x, y, z = [], [], []
        for n in range(input_data.shape[1]):
            x.extend(np.linspace(start=input_data[0][n], stop=input_data[0][n+1], num=ratio))
            y.extend(np.linspace(start=input_data[1][n], stop=input_data[1][n+1], num=ratio))
            z.extend(np.linspace(start=input_data[2][n], stop=input_data[2][n+1], num=ratio))
        return np.array([x, y, z])

def sliding_window(input_data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    # Number of windows
    n = int((len(input_data) - window_size) / stride + 1)
    return np.array([input_data[i * stride:i * stride + window_size] for i in range(n)])

def download_url():
    pass
