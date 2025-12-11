import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=True)  #训练数据集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=True)  #测试数据集

# print(len(train_set))  #训练集样本数
# print(test_set.classes)
#
# print(test_set.classes[train_set[0][1]])
# img , label = train_set[0]
# img.show()
# print(train_set[0])
#print(test_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img, label = train_set[i]
    writer.add_image("CIFAR10_10", img, i)

writer.close()