import torch.nn as nn
import torch.nn.functional as F

# CNN_01 structure refer to FallAllD paper
class CNN_01(nn.Module):
    def __init__(self, input_length, output_size):
        super(CNN_01, self).__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2, padding=0)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        fc_input_size = 32 * ((input_length - 6) // 2)
        self.fc1 = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        
        return out

   
# main
# for debug and watch model structure
if __name__ == '__main__':
    from torchinfo import summary
    
    model = CNN_01()
    summary(model, (8, 1, 260, 3))