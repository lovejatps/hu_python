import os
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label_file = label_file
        self.path = os.path.join(self.root_dir, self.label_file)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.img_path[index])
        return img_path,self.label_file



    def __len__(self):
        return len(self.img_path)

root_dir ='dataset/train'
ants_file = 'ants'
bees_file = 'bees'

ants_dataset = MyDataset(root_dir, ants_file, transform=torchvision.transforms.ToTensor())
bees_dataset = MyDataset(root_dir, bees_file, transform=torchvision.transforms.ToTensor())
image, label = ants_dataset[0]
Image.open(image).show()
print("ants_data:{}".format(ants_dataset.__len__()))
print("bees_data:{}".format(bees_dataset.__len__()))




