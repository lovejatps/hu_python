import torch
from torch.utils.data import Dataset
from PIL import Image
import os


#img_path = "dataset/train/ants"
root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

if(torch.cuda.is_available()):
    print("Hello, World!")


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, index):
       img_name = self.img_path[index]
       img_itme_path = os.path.join(self.path, img_name)
       img = Image.open(img_itme_path)
       label = self.label_dir
       return img, label

    def __len__(self):
        return len(self.img_path)


ants_data = Mydata(root_dir, ants_label_dir)
img, label = ants_data[0]
print(img)
print(label)
img.show()
