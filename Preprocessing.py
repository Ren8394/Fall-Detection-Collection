import numpy as np
import pandas as pd

from pathlib import Path


def preprocessing_FallAllD(loadfile_path, savefile_path, sensor, location, sampling_rate):
    # read data
    # columns = ['SubjectID', 'Device', 'Activity', 'Acc', 'Gyr']
    df = pd.read_pickle(loadfile_path)

    # drop unused columns
    if sensor == ['Acc', 'Gyr']:
        pass
    elif sensor == ['Acc']:
        df = df.drop(columns=['Gyr'])
    elif sensor == ['Gyr']:
        df = df.drop(columns=['Acc'])
    else:
        raise ValueError('sensor must be Acc or Gyr')

    # drop unused rows
    if location == ['Waist', 'Wrist']:
        pass
    elif location == ['Waist']:
        df = df[df['Device'] == 'Waist']
    elif location == ['Wrist']:
        df = df[df['Device'] == 'Wrist']
    else:
        raise ValueError('location must be Waist or Wrist')

    # resampling
    df['Acc'] = df['Acc'].apply(lambda x: x[sorted(
        [len(x)-sampling_rate*(i+1) for i in range(13 * sampling_rate)]), :])

    # one-hot encoding
    df['Activity'] = df['Activity'].apply(
        lambda x: [1, 0] if x == 'Fall' else [0, 1])

    # reset index
    df = df.reset_index(drop=True, inplace=False)
    df.to_pickle(savefile_path)
