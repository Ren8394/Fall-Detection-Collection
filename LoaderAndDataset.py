import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.optim import Adam,SGD
from torch.utils.data import Dataset, DataLoader

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if param.requires_grad==True:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

def load_model(config, model):

    # check and select device
    device = torch.device(f"cuda:{config['others']['gpu']}" if torch.cuda.is_available() else 'cpu')

    # loss fuction
    criterions = {
        'ce'      : nn.CrossEntropyLoss(),
        'mse'     : nn.MSELoss(),
    }
    criterion = criterions[config['model']['loss_fuction']]

    # optimizer
    optimizers = {
        'Adam'    : Adam(model.parameters(), lr=float(config['hyperparameters']['learning_rate']), weight_decay=0),
        'SGD'     : SGD(model.parameters(),lr=float(config['hyperparameters']['learning_rate']), weight_decay=0)
    }
    optimizer = optimizers[config['model']['optimizer']]

    # initial scheduler
    epoch = 0
    best_loss = 1e6
    model.apply(weights_init)

    return model, epoch, best_loss, optimizer, criterion, device

def load_data(config, data_path):
    # read data and split train/val
    data_df = pd.read_pickle(data_path)
    train_df, val_df = train_test_split(data_df, test_size=0.2)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # create dataset
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)

    data_loader = { 
        'train':DataLoader(train_dataset, batch_size=int(config['hyperparameters']['batch_size']), shuffle=True, num_workers=4, pin_memory=False),
        'val'  :DataLoader(val_dataset, batch_size=int(config['hyperparameters']['batch_size']), num_workers=4, pin_memory=False)
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
            acc = torch.tensor(acc, dtype=torch.float32)
            gyr = torch.tensor(gyr, dtype=torch.float32)
            acc = acc.unsqueeze(0)
            gyr = gyr.unsqueeze(0)
            x = torch.cat((acc, gyr), dim=2)
        elif 'Acc' in self.df.columns.tolist():
            acc = self.df.loc[idx, 'Acc']
            acc = torch.tensor(acc, dtype=torch.float32)
            acc = acc.unsqueeze(0)
            x = acc
        elif 'Gyr' in self.df.columns.tolist():
            gyr = self.df.loc[idx, 'Gyr']
            gyr = torch.tensor(gyr, dtype=torch.float32)
            gyr = gyr.unsqueeze(0)
            x = gyr
        else:
            raise ValueError('sensor must be Acc or Gyr')


        label = self.df.loc[idx, 'Activity']
        y = torch.tensor(label, dtype=torch.float32)
        
        return x, y