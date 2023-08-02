import torch.nn as nn
import torch.nn.functional as F

# LSTM_01 structure refer to FallAllD paper
class LSTM_01(nn.Module):
    def __init__(self):
        super(LSTM_01, self).__init__()
        self.lstmBlock1 = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.lstmBlock2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        
        x = x.squeeze(1)
        out, _ = self.lstmBlock1(x)
        out, _ = self.lstmBlock2(out)
        out = out[:, -1, :]
        out = self.fc1(out)

        return out

# main
# for debug and watch model structure
if __name__ == '__main__':
    from torchinfo import summary
    
    model = LSTM_01()
    summary(model, (8, 1, 260, 3))