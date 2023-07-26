import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

MAX = np.iinfo(np.int16).max
EPSILON = np.finfo(float).eps

class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, criterion, device, 
                 loader, writer, output_path, save_results, config):
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
        filename = f"{config['model']['name']}_{config['data']['name']}" \
            f"_{config['model']['loss_fuction']}" \
            f"_epochs{config['hyperparameters']['epochs']}" \
            f"_batch{config['hyperparameters']['batch_size']}" \
            f"_lr{config['hyperparameters']['learning_rate']}"
        self.model_path = output_path.joinpath(f"{filename}.pth")
        self.results_path = output_path.joinpath(f"{filename}.csv")
        self.save_results = save_results
        self.config = config

        self.train_loss = 0        
        self.val_loss = 0
         
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

        self.model.train()
         
        for x, y in self.loader['train']:
            self._train_step(x, y)
            
        self.train_loss /= len(self.loader['train'])

        print(f'Epoch {self.epoch + 1:03d}/{self.epochs} | Train Loss: {self.train_loss:.4f}')

    def _val_step(self, x, y):
        
        device = self.device
        
        x = x.to(device)
        y = y.to(device)
        
        pred = self.model(x)
        loss = self.criterion(pred, y)
        
        self.val_loss += loss.item()

    def _val_epoch(self):
        pass
        self.val_loss = 0
     
        self.model.eval()

        for x, y in self.loader['val']:
            self._val_step(x, y)

        self.val_loss /= len(self.loader['val'])
       
        if self.best_loss > self.val_loss:
            self.save_checkpoint()
            self.best_loss = self.val_loss

    def save_checkpoint(self):

        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        torch.save(state_dict, self.model_path)

    def train(self):
        
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()
            
            plot_name = f"{self.config['model']['name']}_{self.config['data']['name']}" \
                        f"_{self.config['model']['loss_fuction']}" \
                        f"_epochs{self.config['hyperparameters']['epochs']}" \
                        f"_batch{self.config['hyperparameters']['batch_size']}" \
                        f"_lr{self.config['hyperparameters']['learning_rate']}"

            self.writer.add_scalars(plot_name, {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(plot_name, {'val': self.val_loss},self.epoch)
                                
            self.epoch += 1

    def test(self):
        
        # load model
        self.model.eval()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])

        device = self.device    
        self.model.to(device)

        # Confusion matrix
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for _, (x, y) in enumerate(self.loader['val']):
                x = x.to(device)
                y = y.to(device)

                y_pred.append(F.softmax(self.model(x), dim=1).argmax(dim=1).cpu().numpy())
                y_true.append(y.argmax(dim=1).cpu().numpy())
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        bal_acc = (sensitivity + specificity) / 2

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Bal. Acc: {bal_acc:.4f}')
        if self.output_path.joinpath('images', 'confusion_matrix.png').exists():
            plt.savefig(self.output_path.joinpath('images', f"confusion_matrix_{int(time.time())}.png"))
        else:
            plt.savefig(self.output_path.joinpath('images', f"confusion_matrix.png"))
        plt.show()
        plt.close()
    