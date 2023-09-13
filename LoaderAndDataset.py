import importlib
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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
    # import dataset
    module = importlib.import_module("datasets")
    CustomDataset = getattr(module, args.dataset)

    # CustomDataset(split: str = "train",sr: int = 20,window: set = (10, 0.5), location: list = ["Wrist"],  download: bool = False)
    train_dataset = CustomDataset(split='train', sr=args.sampling_rate, window=(args.duration, args.overlap), location=args.location, download=False)
    val_dataset = CustomDataset(split='val', sr=args.sampling_rate, window=(args.duration, args.overlap), location=args.location, download=False)
    test_dataset = CustomDataset(split='test', sr=args.sampling_rate, window=(args.duration, args.overlap), location=args.location, download=False)

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False ,num_workers=4, pin_memory=False),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    }

    return data_loader
