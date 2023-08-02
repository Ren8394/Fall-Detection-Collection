import torch.nn as nn
import torch.nn.functional as F

# CNN_01 structure refer to FallAllD paper
class CNNLSTM_01(nn.Module):
    def __init__(self):
        super(CNNLSTM_01, self).__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2, padding=0)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 1), stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.lstmBlock1 = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        self.lstmBlock2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 2)
        

    def forward(self, x):
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        out = out.squeeze(3)
        out = out.permute(0, 2, 1)
        out, _ = self.lstmBlock1(out)
        out, _ = self.lstmBlock2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out

   
# main
# for debug and watch model structure
if __name__ == '__main__':
    from torchinfo import summary
    
    model = CNNLSTM_01()
    summary(model, (8, 1, 260, 3))