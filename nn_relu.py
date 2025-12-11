import torch
import torch.nn as nn
import torchvision
from torch.distributed.elastic.rendezvous.etcd_server import stop_etcd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False , download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


####非线性变化神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x

model = Net()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    inputs, labels = data
    writer.add_images(    "input_image", inputs, step)
    output = model(inputs)
    writer.add_images( "output_image", output, step)
    step += 1

writer.close()