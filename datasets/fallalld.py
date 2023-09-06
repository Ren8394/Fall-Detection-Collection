import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple


class FallAllD(Dataset):
    
    base_folder = Path.cwd().joinpath("TMP")
    _RESOURCES = {
        "train": ("FallAllD_train.pkl", "202308311431"),
        "val": ("FallAllD_val.pkl", "202308311431"),
        "test": ("FallAllD_test.pkl", "202308311431")
    }


    def __init__(self, split: str = "train", download: bool = False) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        pass
