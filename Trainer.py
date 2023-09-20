import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

MAX = np.iinfo(np.int16).max
EPSILON = np.finfo(float).eps

class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, criterion, device,
                 loader, writer, output_path, save_results, args):
        self.epoch = epoch
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.writer = writer

        self.output_path = output_path
        self.filename = \
            f"{args.model}_{args.dataset}_{''.join(args.location)}" \
            f"_{args.loss_function}" \
            f"_epochs{args.epochs}" \
            f"_batch{args.batch_size}" \
            f"_lr{args.lr}"
        self.output_model_path = output_path.joinpath(f"{self.filename }.pth")
        self.results_path = output_path.joinpath(f"{self.filename }.csv")
        self.save_results = save_results
        self.args = args

        self.train_loss = 0
        self.val_loss = 0

        self.early_stopping = epochs // 10
        self.early_stopping_counter = 0

    def _train_step(self, x, y):

        device = self.device

        x = x.to(device)
        y = y.to(device)

        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)

        self.train_loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def _train_epoch(self):

        self.train_loss = 0

        progress = tqdm(total=len(self.loader['train']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | Train', unit='step', leave=False)
        self.model.train()

        for x, y in self.loader['train']:
            self._train_step(x, y)
            progress.update(1)

        progress.close()
        self.train_loss /= len(self.loader['train'])    

    def _val_step(self, x, y):

        device = self.device

        x = x.to(device)
        y = y.to(device)

        pred = self.model(x)
        loss = self.criterion(pred, y)

        self.val_loss += loss.item()

    def _val_epoch(self):
        
        self.val_loss = 0

        progress = tqdm(total=len(self.loader['val']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | Valid', unit='step', leave=False)
        self.model.eval()

        for x, y in self.loader['val']:
            self._val_step(x, y)
            progress.update(1)

        progress.close()
        self.val_loss /= len(self.loader['val'])

        if self.best_loss > self.val_loss:
            self.save_checkpoint()
            self.best_loss = self.val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

    def save_checkpoint(self):

        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(state_dict, self.output_model_path)

    def train(self):

        print("---------------------------------")
        while self.epoch < self.epochs and self.early_stopping_counter < self.early_stopping:
            self._train_epoch()
            self._val_epoch()

            self.writer.add_scalars(self.filename, {'train': self.train_loss}, self.epoch)
            self.writer.add_scalars(self.filename, {'val': self.val_loss}, self.epoch)

            self.epoch += 1
        
        if self.early_stopping_counter == self.early_stopping:
            print(f'Early stopping at epoch {self.epoch}')
        print("---------------------------------\n\n")

    def test(self):

        # load model
        self.model.eval()
        checkpoint = torch.load(self.output_model_path)
        self.model.load_state_dict(checkpoint['model'])

        device = self.device
        self.model.to(device)

        # Confusion matrix
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for _, (x, y) in enumerate(self.loader['test']):
                x = x.to(device)
                y = y.to(device)

                y_pred.append(F.softmax(self.model(x), dim=1).argmax(dim=1).cpu().numpy())
                y_true.append(y.argmax(dim=1).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Caluculate F1 score, precision, recall, accuracy
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        # Print confusion matrix
        sns.heatmap(
            cm, 
            annot=True,
            annot_kws={'fontsize':14, 'fontweight':'bold'},
            fmt='d', 
            xticklabels=['ADL', 'Fall'], 
            yticklabels=['ADL', 'Fall'], 
            cmap='coolwarm', 
            linecolor='white',
            linewidths=1,
            cbar=False
        )
        plt.ylabel('Actual Labels', fontsize=12)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(r"$\bf{"+ self.args.model + "\_" + self.args.dataset + "}$" + "\n"+ f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")

        # Save results
        self.output_path.joinpath('images').mkdir(parents=True, exist_ok=True)
        if self.output_path.joinpath('images', f"{self.filename}_cm_0.png").exists():
            number_of_files = len(list(self.output_path.joinpath('images').glob(f"{self.filename}_cm_*.png")))
            plt.savefig(self.output_path.joinpath('images', f"{self.filename}_cm_{number_of_files+1}.png"))
        else:
            plt.savefig(self.output_path.joinpath('images', f"{self.filename}_cm_0.png"))
        plt.close()
