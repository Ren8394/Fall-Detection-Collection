import argparse
import os
import pandas as pd
import random
import sys
import torch
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from Preprocessing import preprocessing
from LoaderAndDataset import load_model, load_data
from Trainer import Trainer

# fix random
SEED = 4444
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='FallAllD')
    # preprocessing
    parser.add_argument('--location', type=str, nargs='+', default=['Waist'])
    parser.add_argument('--sampling_rate', type=int, default=20)                # sampling rate in Hz
    parser.add_argument('--duration', type=int, default=10)                     # window size in seconds
    parser.add_argument('--overlap', type=float, default=0.5)                   # overlap ratio
    # model
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--version', type=str, default='01')
    parser.add_argument('--loss_function', type=str, default='ce')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--status', type=str, default='train')
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    # etc
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_results', type=str, default='False')

    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    # get current path
    print('* current working directory =', Path.cwd())
    print('* random seed =', SEED)

    # get parameter
    args = get_args()

    print('* model name =', args.model)
    print('* status =', args.status)
    print('* learning rate =', args.lr)

    # data path
    preprocessing(
        dataset=args.dataset,
        device_location=args.location,
        sampling_rate=args.sampling_rate,
        duration=args.duration,
        overlap=args.overlap
    )

    # load model
    exec(f"from models.{args.model} import {args.model}_{args.version} as model")
    model = model(input_length=int(args.sampling_rate * args.duration), output_size=2)
    model, epoch, best_loss, optimizer, criterion, device = load_model(args, model)

    # tensorboard
    writer = SummaryWriter(
        f"./logs/"
        f"{args.model}_{args.dataset}_{''.join(args.location)}"
        f"_{args.loss_function}"
        f"_epochs{args.epochs}"
        f"_batch{args.batch_size}"
        f"_lr{args.lr}"
    )

    data_loader = load_data(args, data_path=Path.cwd().joinpath('datasets', 'processed', f"{args.dataset}-Processed.pkl"))

    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, criterion, device,
                      data_loader, writer, Path.cwd().joinpath('results'), True, args)

    try:
        if args.status == 'train':
            Trainer.train()
        Trainer.test()
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }
        filename = \
            f"{args.model}_{args.dataset}_{''.join(args.location)}" \
            f"_{args.loss_function}" \
            f"_epochs{args.epochs}" \
            f"_batch{args.batch_size}" \
            f"_lr{args.lr}"
        checkpoint_path = Path.cwd().joinpath('results').joinpath(f"{filename}.pth")
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
