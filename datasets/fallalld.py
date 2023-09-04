import numpy as np
import pandas as pd

from typing import Tuple, Any
from torch.utils.data import Dataset

class fallalld(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        pass
