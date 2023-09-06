import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Tuple

from .utils import resample, sliding_window

class FallAllD(Dataset):
    
    base_folder = Path.cwd().joinpath(".tmp", "data")
    _RESOURCES = {
        "raw": ("FallAllD.pkl"),
        "train": ("FallAllD_train.pkl"),
        "val": ("FallAllD_val.pkl"),
        "test": ("FallAllD_test.pkl")
    }

    def __init__(
        self, 
        split: str = "train", 
        download: bool = False
    ) -> None:
        
        super().__init__()
        self.split = split
        self.download = download

        if not self._check_dataset_exists():
            raise RuntimeError(
                "FallAllD Dataset not found. You can download it from " +\
                "https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living" +\
                f"extract its and execute python file then move FallAllD.pkl to the {self.base_folder} folder "
            )
        
        # sampling rate in Hz
        self.sr = 20
        # window size in points (seconds * sampling rate)          
        self.window_size = int(10 * self.sr)
        # window stride in points (window size * overlap ratio)
        self.window_stride = int(self.window_size * 0.5)

        # process dataset
        self.raw = pd.read_pickle(self.base_folder.joinpath(self._RESOURCES["raw"][0]))
        self._process()


    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        pass

    def _check_dataset_exists(self) -> bool:
        
        if not self.base_folder.exists():
            self.base_folder.mkdir(parents=True, exist_ok=True)
        else:
            if not self.base_folder.joinpath(self._RESOURCES["raw"][0]).exists():
                return False
        return True

    def _process(self) -> pd.DataFrame:
        
        self.raw = self.raw.dropna()
        # drop unused columns
        # keep column SubjectID, Device, ActivityID, TrialNo, and Acc	
        self.raw = self.raw.drop(columns=["Gyr", "Mag", "Bar"])
        self.raw = self.raw.reset_index(drop=True)

        # resample
        self.raw["Acc"] = self.raw["Acc"].apply(lambda x: resample(x, 238, self.sr))
        # sliding window
        for i, row in self.raw.iterrows():
            # ADL
            if row["ActivityID"] < 100:
                wins = sliding_window(row["Acc"], self.window_size, self.window_stride)
                for i in wins.shape[0]:
                    self.raw.loc[len(self.raw)] = [
                        row["SubjectID"], 
                        row["Device"], 
                        row["ActivityID"],
                        f"{row['TrialNo']}-{i+1}",
                        wins[i]
                    ]
                # delete row
                self.raw = self.raw.drop(index=i)
        # sort by SubjectID, Device, ActivityID, TrialNo
        self.raw = self.raw.sort_values(by=["SubjectID", "Device", "ActivityID", "TrialNo"])



                   