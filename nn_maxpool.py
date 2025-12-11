import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

##准备数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=False  , download=True,
                                           transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


###最大池化网络   最大池化层作用，保留主要特征，去除不重要的特征，减少计算量与提高网络的鲁棒性
class MaxPoolNet(nn.Module):
    def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.maxpool = nn.MaxPool2d(3,ceil_mode=True)

    def forward(self, x):
        output = self.maxpool(x)
        return output

writer = SummaryWriter("logs")
model = MaxPoolNet()
step = 0
for data in dataloader:
    img, label = data
    writer.add_images("input", img,step)
    output = model(img)
    writer.add_images("output", output, step)
    step += 1

writer.close()




