import numpy as np
import pandas as pd

from pathlib import Path

dataset_sampling_rate = {
    'FallAllD': {
        'Waist': 238,
        'Wrist': 238,
        'Neck': 238,
    },
    'UMAFall': {
        'Waist': 20,
        'Wrist': 20,
        'Ankle': 20,
        'Chest': 20,
        'RightPocket': 200,
    },
    'SisFall': {
        'Waist': 200,
    }
}

def sliding_window(x, window_size, overlap):
    # Step size
    step = int(window_size * overlap)
    # Number of windows                            
    n = int((len(x) - window_size) / step + 1)
    x = np.array([x[i * step:i * step + window_size] for i in range(n)])
    return x

def resampling(x, original_sampling_rate, target_sampling_rate, target_duration):
    # Down-sampling
    if original_sampling_rate > target_sampling_rate: 
        step = int(np.floor(original_sampling_rate / target_sampling_rate))
        x = x[::-step, :]
    # Up-sampling
    elif original_sampling_rate < target_sampling_rate: 
        step = int(np.ceil(target_sampling_rate / original_sampling_rate))
        x = np.repeat(x, step, axis=0)
    
    # Padding
    if len(x) > int(target_duration * target_sampling_rate):
        x = x[-int(target_duration * target_sampling_rate):, :]
    elif len(x) < int(target_duration * target_sampling_rate):
        x = np.concatenate((x, np.zeros((int(target_duration * target_sampling_rate) - len(x), 3))), axis=0)
    return x

def preprocessing(dataset, device_location, sampling_rate, duration, overlap):
    # Check dataset
    if not dataset in list(dataset_sampling_rate.keys()):
        raise ValueError(f"dataset must be {list(dataset_sampling_rate.keys())}, but got {dataset}")
    
    # Path for loading data and saving data
    loadfile_path = Path.cwd().joinpath('dataset', 'processed', f"{dataset}-Preliminary.pkl")
    savefile_path = Path.cwd().joinpath('dataset', 'processed', f"{dataset}-Processed.pkl")

    # Preprocessing
    # Read data
    # Columns = ['SubjectID', 'Device', 'Activity', 'Acc']
    df = pd.read_pickle(loadfile_path)

    # Sliding window (Only ADL data)
    list_subjectID = []
    list_Device = []
    list_Activity = []
    list_Acc = []
    for _, row in df.iterrows():
        if row['Activity'] == 'Fall':
            list_subjectID.append(row['SubjectID'])
            list_Device.append(row['Device'])
            list_Activity.append(row['Activity'])
            list_Acc.append(resampling(row['Acc'], dataset_sampling_rate[dataset][row['Device']], sampling_rate, duration))
        else:
            for win in sliding_window(row['Acc'], dataset_sampling_rate[dataset][row['Device']] * duration, overlap):
                list_subjectID.append(row['SubjectID'])
                list_Device.append(row['Device'])
                list_Activity.append(row['Activity'])
                list_Acc.append(resampling(win, dataset_sampling_rate[dataset][row['Device']], sampling_rate, duration))

    del df
    df = pd.DataFrame({'SubjectID': list_subjectID, 'Device': list_Device, 'Activity': list_Activity, 'Acc': list_Acc})

    # Drop unused rows
    if all(x in list(dataset_sampling_rate[dataset].keys()) for x in device_location):
        df = df[df['Device'].isin(device_location)]
    else:
        raise ValueError(f'device_location must be {list(dataset_sampling_rate[dataset].keys())}, but got {device_location}')
    
    # One-hot encoding
    df['Activity'] = df['Activity'].apply(lambda x: [1, 0] if x == 'Fall' else [0, 1])
    
    # Reset index
    df = df.reset_index(drop=True, inplace=False)
    df.to_pickle(savefile_path)

if __name__ == '__main__':
    
    preprocessing(
        dataset='FallAllD',
        device_location=['Waist'],
        sampling_rate=20,
        duration=10,
        overlap=0.5
    )