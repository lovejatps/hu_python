import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class HuxnNet(torch.nn.Module):
    def __init__(self):
        super(HuxnNet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3,32,5,padding=2)  ## 卷积层32 filters of size 5x5
        # self.maxpool1 = torch.nn.MaxPool2d(2)   ## 最大池化层 2x2
        # self.conv2 = torch.nn.Conv2d(32,32,5,padding=2)  ## 卷积层  32 filters of size 5x5
        # self.maxpool2 = torch.nn.MaxPool2d(2)  ## 最大池化层 2x2
        # self.conv3 = torch.nn.Conv2d(32,64,5,padding=2)  ## 卷积层 64 filters of size 5x5
        # self.maxpool3 = torch.nn.MaxPool2d(2)  ## 最大池化层 2x2
        # self.flatten = torch.nn.Flatten()  ## 展平层
        # self.linear1 = torch.nn.Linear(1024,64)  #线性层     全连接层 64 neurons
        # self.linear2 = torch.nn.Linear(64,10)  #线性层     全连接层 10 neurons

        self.model1 = torch.nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),  ## 卷积层32 filters of size 5x5
            nn.MaxPool2d(2),                                                ## 最大池化层 2x2
            nn.Conv2d(32,32,5,padding=2),  ## 卷积层  32 filters of size 5x5
            nn.MaxPool2d(2),                                                ## 最大池化层 2x2
            nn.Conv2d(32,64,5,padding=2),                                   ## 卷积层 64
            nn.MaxPool2d(2),                                               ## 最大池化层 2x2
            nn.Flatten(),                                                 ## 展平层
            nn.Linear(1024,64),                                             #线性层     全连接层 64 neurons
            nn.Linear(64,10),                                                #线性层     全连接层 10 neurons
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

#创建模型
huxnnet = HuxnNet()
print(huxnnet)
writer = SummaryWriter('logs')

##准备数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader_test =torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
print("测试数据集：{}".format(len(dataloader_test)))
print("训练数据集：{}".format(len(dataset)))

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
datatest_loader = DataLoader(dataloader_test, batch_size=64, shuffle=True)

global_step=0
global_step_test=0
loss_fn = torch.nn.CrossEntropyLoss()   ## 创建损失函数
learning_rate = 1e-2   ## 学习率  1e-2 = 1 * 10^-2 = 0.01
optimizer = torch.optim.SGD(huxnnet.parameters(), lr=learning_rate)  ## 创建优化器

##训练模型
epoch = 20
for epoch in range(epoch):
    huxnnet.train()  ### 开启dropout等激活函数
    for img,target in dataloader:
        if epoch == 0:
            writer.add_images('inputimages', img   , global_step)
        output = huxnnet(img)
        loss = loss_fn(output, target)
        if global_step % 100 == 0:
            writer.add_scalar('训练:{} 轮'.format(epoch), loss, global_step)
            print('训练：epoch:{}, step:{}, loss:{}'.format(epoch,global_step,loss))
        ###优化器优化模型
        optimizer.zero_grad()  ## 梯度清零
        loss.backward()     ## 损失函数反向传播
        optimizer.step()     ## 优化器更新参数
        global_step+=1

    ####测试模型步骤

    total_test_loss = 0
    huxnnet.eval()  ### 关闭dropout等激活函数
    with  torch.no_grad():  ### 关闭梯度计算
        for data  in datatest_loader:
            imgs, target = data
            output = huxnnet(imgs)
            loss = loss_fn(output, target) 
            total_test_loss += loss.item()
    print('测试：epoch:{}, loss:{}'.format(epoch,total_test_loss))
    writer.add_scalar('测试epoch', total_test_loss, epoch)
    torch.save(huxnnet.state_dict(), 'huxn{}.pth'.format(epoch))

writer.close()