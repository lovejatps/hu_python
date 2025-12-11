import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

### 卷积网络
class JatpsNet(torch.nn.Module):
    def __init__(self):
        super(JatpsNet, self).__init__()
        ### in_channels 输入通道数,【颜色图片为3】
        # out_channels 输出通道数,【输出特征图的通道数，对应券积核的个数】
        # kernel_size 卷积核大小,
        # stride=1 横向、纵向步长,
        # padding=0 填充【图片左右、上下边缘填充的像素数】 ,
        # padding_mode='zeros' 填充数值,
        # dilation=1 膨胀率,
        # groups=1 组数,
        # bias=True 是否使用偏置项
        self.conv1 = torch.nn.Conv2d(3, 6, 3,stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = JatpsNet()

writer = SummaryWriter("logs")
stape = 0
for data in dataloader:
    imgs, targets = data
    #imgs = imgs.view(-1,3,64,64)
    output = model(imgs)
    print(output.shape)
    writer.add_images("intput", imgs, stape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, stape)
    stape += 1

writer.close()