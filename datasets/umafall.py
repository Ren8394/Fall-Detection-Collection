import numpy as np
import pandas as pd
import re
import requests
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Tuple
from zipfile import ZipFile

from .utils import resample, sliding_window, ensure_type, check_md5

class UMAFall(Dataset):
    
    base_folder = Path.cwd().joinpath(".tmp", "data")
    _number_of_files = int(746)
    _RESOURCES = {
        "compress": ("UMAFall_Dataset.zip", "5ea9f4cb6277272d2f5f3f94e8ed456f"),
        "raw": ("UMAFall.pkl", "..."),
        "train": ("UMAFall_train.pkl", "..."),
        "val": ("UMAFall_val.pkl", "..."),
        "test": ("UMAFall_test.pkl", "...")
    }

    def __init__(
        self, 
        split: str = "train",
        sr: int = 20,
        window: set = (10, 0.5), 
        location: list = ["Wrist"],   
        download: bool = False
    ) -> None:
        
        super().__init__()
        self.download = download
        self.location = location
        # sampling rate in Hz
        self.sr = sr
        # window size in points (seconds * sampling rate)          
        self.window_size = int(window[0] * sr)
        # window stride in points (window size * overlap ratio)
        self.window_stride = int(self.window_size * window[1])

        self.base_folder.mkdir(parents=True, exist_ok=True)
        # check dataset zip file exists, and if exists, check MD5 checksum
        if not self.base_folder.joinpath(self._RESOURCES["compress"][0]).exists():
            self._download()
        elif not check_md5(self.base_folder.joinpath(self._RESOURCES["compress"][0]), self._RESOURCES["compress"][1]):
                self._download()

        self.base_folder.joinpath("UMAFall_Dataset").mkdir(parents=True, exist_ok=True)
        # check dataset folder has been extracted correctly
        if len(list(self.base_folder.joinpath("UMAFall_Dataset").glob("*.csv"))) != self._number_of_files:
            self._extract()

        # process dataset
        if split == "train":
            self.preprocess()

        # load processed dataset
        self.df = pd.read_pickle(self.base_folder.joinpath(self._RESOURCES[split][0]))

    def __len__(self) -> int:

        return len(self.df)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        
        acc = self.df.loc[idx, "Acc"]
        acc = torch.tensor(ensure_type(acc), dtype=torch.float32)
        acc = acc.unsqueeze(0)
        x = acc

        label = self.df.loc[idx, "Fall/ADL"]
        y = torch.tensor(label, dtype=torch.float32)

        return x, y

    def _download(self) -> None:
        
        response = requests.get("https://figshare.com/ndownloader/files/11826395", stream=True)
        download_filepath = self.base_folder.joinpath(self._RESOURCES['compress'][0])
        
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading UMAFall Dataset") as t:
            with download_filepath.open('wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    t.update(len(chunk))
                    f.write(chunk)

        # check MD5 checksum
        if not check_md5(download_filepath, self._RESOURCES["compress"][1]):
            raise AssertionError("MD5 mismatch! File might be corrupted.")

    def _extract(self) -> None:
        
        zip_filepath = self.base_folder.joinpath(self._RESOURCES["compress"][0])
        with ZipFile(zip_filepath, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            with tqdm(total=len(all_files), desc="Extracting") as t:
                for member in all_files:
                    zip_ref.extract(member, self.base_folder)
                    t.update(1)

    def _get_filename_info(self, filename:str) -> Tuple[int, list, str, int]:
        # get subject id
        id = re.search(r"Subject_(\d+)", filename).group(1)
        # get ADL or Fall
        adl_or_fall = re.search(r"_(Fall|ADL)_", filename).group(1)
        adl_or_fall = [1, 0] if adl_or_fall == "ADL" else [0, 1]
        # get action description
        description = re.search(r"_(Fall|ADL)_(\D+?)_\d", filename).group(2).replace('_', '')
        # get trail No.
        trialNo = re.search(r"_(\d+)_\d{4}-\d{2}-\d{2}", filename).group(1)
        
        return id, adl_or_fall, description, trialNo

    def _merge_csv_to_pkl(self) -> None:

        sensor_ID = {
            0: "RightPocket",
            1: "Chest",
            2: "Waist",
            3: "Wrist",
            4: "Ankle"
        }

        list_id, list_device, list_activity, list_trial, list_fall_adl, list_acc = [], [], [], [], [], []
        # merge all csv files into one pkl file, except for subject 13
        # column name -> SubjectID, Device, ActivityID, TrialNo, Fall/ADL, and Acc
        all_files = list(self.base_folder.joinpath("UMAFall_Dataset").glob("*.csv"))
        for file in all_files:
            # UMFall_Subject_ID_ADLorFall_Description_TrailNo_date_time.csv
            id, adl_or_fall, description, trialNo = self._get_filename_info(str(file.stem))

            if id == "13":
                continue
            # read csv file
            df = pd.read_csv(file, header=0, skiprows=40, sep=';')
            df = df.iloc[: , :-1]
            df.columns = ["Time", "Sample No", "X-Axis", "Y-Axis", "Z-Axis", "Sensor Type", "Sensor ID"]

            for sensor_id in sensor_ID:
                device = sensor_ID[sensor_id]
                acc = df.loc[(df["Sensor Type"] == 0) & (df["Sensor ID"] == sensor_id), ["X-Axis", "Y-Axis", "Z-Axis"]].to_numpy()
                # append to data list
                list_id.append(np.uint8(id))
                list_device.append(device)
                list_activity.append(description)
                list_trial.append(np.uint8(trialNo))
                list_fall_adl.append(adl_or_fall)
                list_acc.append(acc)

            df_UMAFall = pd.DataFrame(
                list(zip(list_id, list_device, list_activity, list_trial, list_fall_adl, list_acc)), 
                columns=["SubjectID", "Device", "Activity", "TrialNo", "Fall/ADL", "Acc"]
            )
            # Remove row with empty Acc data
            df_UMAFall = df_UMAFall[df_UMAFall["Acc"].map(len) > 0]
            pd.to_pickle(df_UMAFall, self.base_folder.joinpath(self._RESOURCES["raw"][0]))

    def preprocess(self) -> None:
        # check merged csv file exists
        if not self.base_folder.joinpath(self._RESOURCES["raw"][0]).exists():
            # merge all csv files into one pkl file, except for subject 13
            self._merge_csv_to_pkl()
        raw = pd.read_pickle(self.base_folder.joinpath(self._RESOURCES["raw"][0]))
        # drop row with NaN value
        raw = raw.dropna()
        raw = raw.reset_index(drop=True)

        # column name -> SubjectID, Device, Activity, TrialNo, Fall/ADL, and Acc
        del_idx = []
        for i, row in raw.iterrows():
            # resample
            if row["Device"] == "RightPocket":
                row["Acc"] = resample(row["Acc"], 200, self.sr)
            else:
                row["Acc"] = resample(row["Acc"], 20, self.sr)
            # ADL
            if row["Fall/ADL"][0] == 1:
                # sliding window
                for win in sliding_window(row["Acc"], self.window_size, self.window_stride):
                    raw.loc[len(raw)] = [
                        row["SubjectID"], 
                        row["Device"], 
                        row["Activity"],
                        row["TrialNo"],
                        [1, 0],
                        win
                    ]
                # add row index to be deleted
                del_idx.append(i)
            # Fall
            else:
                # column Acc extrct only middle 10 seconds (10 * sr)
                start_cutting = int(np.floor((len(row["Acc"]) - self.window_size) / 2))
                end_cutting = int(start_cutting + self.window_size)
                raw.loc[len(raw)] = [
                    row["SubjectID"], 
                    row["Device"], 
                    row["Activity"],
                    row["TrialNo"],
                    [0, 1],
                    row["Acc"][start_cutting:end_cutting],
                ]
                # add row index to be deleted
                del_idx.append(i)
        # delete row
        raw = raw.drop(del_idx)

        # select device locations
        assert all(loc in list(raw["Device"].unique()) for loc in self.location), \
            f"device_location must be {list(raw['Device'].unique())}, but got {self.location}"
        raw = raw[raw["Device"].isin(self.location)]
        # sort by SubjectID, Device, Activity, TrialNo
        raw = raw.sort_values(by=["SubjectID", "Device", "Activity", "TrialNo"])
        raw = raw.reset_index(drop=True)

        # split dataset
        train, val = train_test_split(raw, test_size=0.2, random_state=4444)
        val, test = train_test_split(val, test_size=0.5, random_state=4444)
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # save dataset
        train.to_pickle(self.base_folder.joinpath(self._RESOURCES["train"][0]))
        val.to_pickle(self.base_folder.joinpath(self._RESOURCES["val"][0]))
        test.to_pickle(self.base_folder.joinpath(self._RESOURCES["test"][0]))
    