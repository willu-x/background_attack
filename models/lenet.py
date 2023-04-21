import torch
import torch.nn as nn
import torch.nn.functional as F
# class LeNet(nn.Module):
    # def __init__(self):
    #     super(LeNet, self).__init__()
    #     act = nn.Sigmoid
    #     self.body = nn.Sequential(
    #         nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
    #         act(),
    #         nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
    #         act(),
    #         nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
    #         act(),
    #         nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
    #         act(),
    #     )
    #     self.fc = nn.Sequential(
    #         nn.Linear(768, 10)
    #     )
    # def forward(self, x):
    #     out = self.body(x)
    #     out=out.reshape(-1)
    #     out = self.fc(out)
    #     return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def createLeNet():
    model = LeNet()
    # model.apply(weights_init)
    return model