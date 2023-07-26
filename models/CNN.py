import torch.nn as nn
import torch.nn.functional as F

# CNN_1 structure refer to FallAllD paper
class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2, padding=0)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * 127, 2)

    def forward(self, x):
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        
        return out

   
# main
# for debug and watch model structure
if __name__ == '__main__':
    from torchsummary import summary
    
    model = CNN_1()
    summary(model, (1, 260, 3))