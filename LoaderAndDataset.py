import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if param.requires_grad == True:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


def load_model(args, model):

    # check and select device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    # loss fuction
    criterions = {
        'ce': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
    }
    criterion = criterions[args.loss_function]

    # optimizer
    optimizers = {
        'adam': Adam(model.parameters(), lr=args.lr, weight_decay=0),
        'sgd': SGD(model.parameters(), lr=args.lr, weight_decay=0)
    }
    optimizer = optimizers[args.optimizer]

    # initial scheduler
    epoch = 0
    best_loss = 1e6
    model.apply(weights_init)

    return model, epoch, best_loss, optimizer, criterion, device


def load_data(args, data_path):
    # read data and split train/val
    data_df = pd.read_pickle(data_path)
    train_df, val_df = train_test_split(data_df, test_size=0.2)
    val_df, test_df = train_test_split(val_df, test_size=0.5)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # create dataset
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=False),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=False)
    }

    return data_loader

##############################################################################################################


class CustomDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # check what sensors are used
        if 'Acc' in self.df.columns.tolist() and 'Gyr' in self.df.columns.tolist():
            acc = self.df.loc[idx, 'Acc']
            gyr = self.df.loc[idx, 'Gyr']
            acc = torch.tensor(self._ensure_type(acc), dtype=torch.float32)
            gyr = torch.tensor(self._ensure_type(gyr), dtype=torch.float32)
            acc = acc.unsqueeze(0)
            gyr = gyr.unsqueeze(0)
            x = torch.cat((acc, gyr), dim=2)
        elif 'Acc' in self.df.columns.tolist():
            acc = self.df.loc[idx, 'Acc']
            acc = torch.tensor(self._ensure_type(acc), dtype=torch.float32)
            acc = acc.unsqueeze(0)
            x = acc
        elif 'Gyr' in self.df.columns.tolist():
            gyr = self.df.loc[idx, 'Gyr']
            gyr = torch.tensor(self._ensure_type(gyr), dtype=torch.float32)
            gyr = gyr.unsqueeze(0)
            x = gyr
        else:
            raise ValueError('sensor must be Acc or Gyr')

        label = self.df.loc[idx, 'Activity']
        y = torch.tensor(label, dtype=torch.float32)

        return x, y

    def _ensure_type(self, data):
        # ensure acc is a numpy array with a supported data type (e.g., float32)
        if isinstance(data, np.ndarray):
            if data.dtype.type is np.object_:
                # change to a supported data type
                data = data.astype(np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32)

        return data
