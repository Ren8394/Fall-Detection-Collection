import argparse
import importlib
import os
import pandas as pd
import random
import sys
import torch
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import date

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
    parser.add_argument('--dataset', type=str, default='UMAFall')
    # preprocessing
    parser.add_argument('--location', type=str, nargs='+', default=['Wrist'])
    parser.add_argument('--sampling_rate', type=int, default=30)                # sampling rate in Hz
    parser.add_argument('--duration', type=int, default=10)                     # window size in seconds
    parser.add_argument('--overlap', type=float, default=0.5)                   # overlap ratio
    # model
    parser.add_argument('--model', type=str, default='CNN')
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
    # get parameter
    args = get_args()

    print("* dataset =", args.dataset)
    print("* model name =", args.model)
    print("* status =", args.status)
    print("* learning rate =", args.lr)

    # load model
    module = importlib.import_module(f"models.{args.model}")
    model_class = getattr(module, f"{args.model}_{args.version}")
    model = model_class(input_length=int(args.sampling_rate * args.duration), output_size=2)
    model, epoch, best_loss, optimizer, criterion, device = load_model(args, model)

    # tensorboard
    writer = SummaryWriter(
        f"./logs/"
        f"{args.model}_{args.dataset}_{''.join(args.location)}/"
        f"{date.today()}/"
        f"epochs{args.epochs}_batch{args.batch_size}_lr{args.lr}_{args.loss_function}"
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
