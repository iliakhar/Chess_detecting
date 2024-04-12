import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, device: torch.device):
        super(ConvNet, self).__init__()
        #21x21x1
        self.layer1 = nn.Sequential(nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        #11x11x12
        self.layer2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        #6x6x24
        self.drop_out = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(6 * 6 * 24, 500)
        self.fc2 = nn.Linear(500, 2)
        # self.sm3 = nn.Softmax(0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        a = out.size(0)
        # out = out.reshape(out.size(0), -1)
        out = out.reshape(1, -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.sm3(out)
        return out
