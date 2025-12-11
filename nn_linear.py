import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



dataset = torchvision.datasets.CIFAR10(root='./data', train=False , download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

model = Linear(3*32*32, 10) 
