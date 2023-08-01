import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.utils import resample

def resampling(x, original_sampling_rate, target_sampling_rate, target_duration):
    
    original_duration = len(x) / original_sampling_rate
    
    step = int(original_sampling_rate / target_sampling_rate)
    x = x[::-step, :]
    
    if original_duration > target_duration:
        x = x[-(target_duration * target_sampling_rate):, :]
    else:
        x = np.concatenate((x, np.zeros((target_sampling_rate * target_duration - len(x), 3))), axis=0)
    return x

def preprocessing_FallAllD(loadfile_path, savefile_path, sensor, location, sampling_rate, duration):
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
        raise ValueError('sensor must be Acc or Gyr, but got {}'.format(sensor))

    # drop unused rows
    if location == ['Waist', 'Wrist']:
        pass
    elif location == ['Waist']:
        df = df[df['Device'] == 'Waist']
    elif location == ['Wrist']:
        df = df[df['Device'] == 'Wrist']
    else:
        raise ValueError('location must be Waist or Wrist, but got {}'.format(location))

    # resampling (Original sampling rate = 238Hz)
    df['Acc'] = df['Acc'].apply(lambda x: resampling(x, 238, sampling_rate, duration))

    # one-hot encoding
    df['Activity'] = df['Activity'].apply(
        lambda x: [1, 0] if x == 'Fall' else [0, 1])

    # reset index
    df = df.reset_index(drop=True, inplace=False)
    df.to_pickle(savefile_path)

def preprocessing_UMAFall(loadfile_path, savefile_path, sensor, location, sampling_rate, duration):
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
        raise ValueError('sensor must be Acc or Gyr, but got {}'.format(sensor))

    # drop unused rows
    if location == ['Waist', 'Wrist']:
        pass
    elif location == ['Waist']:
        df = df[df['Device'] == 'Waist']
    elif location == ['Wrist']:
        df = df[df['Device'] == 'Wrist']
    else:
        raise ValueError('location must be Waist or Wrist, but got {}'.format(location))

    # resampling (Original sampling rate = 20Hz)
    df['Acc'] = df['Acc'].apply(lambda x: resampling(x, 20, sampling_rate, duration))

    # one-hot encoding
    df['Activity'] = df['Activity'].apply(
        lambda x: [1, 0] if x == 'Fall' else [0, 1])

    # reset index
    df = df.reset_index(drop=True, inplace=False)
    df.to_pickle(savefile_path)

def preprocessing_SisFall(loadfile_path, savefile_path, sensor, location, sampling_rate, duration):
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
        raise ValueError('sensor must be Acc or Gyr, but got {}'.format(sensor))

    # drop unused rows
    if location == ['Waist', 'Wrist']:
        pass
    elif location == ['Waist']:
        df = df[df['Device'] == 'Waist']
    elif location == ['Wrist']:
        df = df[df['Device'] == 'Wrist']
    else:
        raise ValueError('location must be Waist or Wrist, but got {}'.format(location))

    # resampling (Original sampling rate = 200Hz)
    df['Acc'] = df['Acc'].apply(lambda x: resampling(x, 200, sampling_rate, duration))

    # one-hot encoding
    df['Activity'] = df['Activity'].apply(
        lambda x: [1, 0] if x == 'Fall' else [0, 1])

    # reset index
    df = df.reset_index(drop=True, inplace=False)
    df.to_pickle(savefile_path)