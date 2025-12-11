import torch.nn as nn
import torchvision
import torch

vgg16_true = torchvision.models.vgg16(pretrained=True)

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

## 在vgg16的最后一层添加一个线性层，输出10类
##vgg16_true.add_module("add_linear", nn.Linear(1000, 10))

## 直接添加一个线性层
vgg16_true.classifier.add_module("add_linear", nn.Linear(4096, 10))

## 或者直接修改vgg16的最后一层
##vgg16_true.classifier[6] = nn.Linear(4096, 10)


##-------------------保存模型方式一-----------------------------------------
## 保存模型
vgg16 = torchvision.models.vgg16(pretrained=False)
## 保存模型结构 、参数
torch.save(vgg16, 'vgg16_model_vgg1.pth')

### 加载模型
model_vgg16 = torch.load('vgg16_model_vgg1.pth')

##-------------------保存模型方式二-----------------------------------------
## 保存模型参数
torch.save(vgg16.state_dict(), 'vgg16_model_vgg2.pth')

### 加载模型参数
vgg16_dict = torchvision.models.vgg16(pretrained=False)
vgg16_dict.load_state_dict(torch.load('vgg16_model_vgg2.pth'))

print(vgg16_dict)

