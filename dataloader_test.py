import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

####准备测试的数据集
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,num_workers=0,drop_last=True)

writer = SummaryWriter("logs")

for epoch in range(2):
    step = 0
    for i,date in enumerate(test_loader):
        img,targets = date
        writer.add_images("Epoch:{}".format(epoch), img, step)
        step += 1

writer.close()