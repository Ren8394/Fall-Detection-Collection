import configparser
import os
import pandas as pd
import random
import sys
import torch
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from LoaderAndDataset import load_model, load_data
from Trainer import Trainer

# fix random
SEED = 4444
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

# main
if __name__ == '__main__':
    # get current path
    print('* current working directory =', Path.cwd())
    print('* random seed =', SEED)

    # get parameter
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('* model name =', config['model']['name'])
    print('* status =', config['model']['status'])
    print('* learning rate =', config['hyperparameters']['learning_rate'])

    # data path
    exec(
        f"from Preprocessing import preprocessing_{config['data']['name']} as preprocess")
    preprocess(
        loadfile_path=Path.cwd().joinpath('datasets', 'processed',
                                          f"{config['data']['name']}-Preliminary.pkl"),
        sensor=config['data']['sensor'].split(','),
        location=config['data']['location'].split(','),
        sampling_rate=int(config['data']['sampling_rate']),
        savefile_path=Path.cwd().joinpath('datasets', 'processed',
                                          f"{config['data']['name']}-Processed.pkl"),
        duration=int(config['data']['duration'])
    )

    # load model
    exec(
        f"from models.{config['model']['name'].split('_')[0]} import {config['model']['name']} as model")
    model = model()
    model, epoch, best_loss, optimizer, criterion, device = load_model(
        config, model)

    # tensorboard
    writer = SummaryWriter(f"./logs/"
                           f"{config['model']['name']}_{config['data']['name']}_{config['model']['loss_fuction']}"
                           f"_epochs{config['hyperparameters']['epochs']}_batch{config['hyperparameters']['batch_size']}_lr{config['hyperparameters']['learning_rate']}")

    data_loader = load_data(config, data_path=Path.cwd().joinpath(
        'datasets', 'processed', f"{config['data']['name']}-Processed.pkl"))

    Trainer = Trainer(model, int(config['hyperparameters']['epochs']), epoch, best_loss, optimizer, criterion, device,
                      data_loader, writer, Path.cwd().joinpath('results'), True, config)

    try:
        if config['model']['status'] == 'train':
            Trainer.train()
        Trainer.test()
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }
        filename = f"{config['model']['name']}_{config['data']['name']}" \
            f"_{config['model']['loss_fuction']}" \
            f"_epochs{config['hyperparameters']['epochs']}" \
            f"_batch{config['hyperparameters']['batch_size']}" \
            f"_lr{config['hyperparameters']['learning_rate']}"
        checkpoint_path = Path.cwd().joinpath(
            'results').joinpath(f"{filename}.pth")
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
