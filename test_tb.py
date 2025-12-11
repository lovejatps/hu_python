from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2


writer = SummaryWriter("logs")

image_path = "dataset/train/ants/0013035.jpg"
image = Image.open(image_path)
print(image.size)
img_array = np.array(image)
writer.add_image("ants_image", img_array, 1 , dataformats='HWC')
print(img_array.shape)

torch_trans = transforms.ToTensor()

### TOTensor
cv_img = cv2.imread(image_path)
tensor_img = torch_trans(cv_img)
writer.add_image("ants_image_tensor", tensor_img, 1, dataformats='CHW')

##Normalize  归一化
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans_norm_img = trans_norm(tensor_img)
writer.add_image("ants_image_tensor", trans_norm_img, 2, dataformats='CHW')
print(tensor_img.shape)

# Resize   图片指定大小的裁剪
trans_resize = transforms.Resize((224, 224))
trans_resize_img = trans_resize(image)
print(trans_resize_img.size)
resize_img = torch_trans(trans_resize_img)
writer.add_image("ants_image_resize", resize_img, 0)

# Compose
trans_resize_2 = transforms.Resize(430)
trans_compose = transforms.Compose([trans_resize_2,torch_trans])
img_resize_2 = trans_compose(image)
writer.add_image("ants_image_compose", img_resize_2, 0)


### RandomCrop    图片随机裁剪
trans_random = transforms.RandomCrop(420)
#trans_random = transforms.RandomCrop(480,320)
trans_compose_2 = transforms.Compose([trans_random,torch_trans])
for i in range(10):
    img_random = trans_compose_2(image)
    writer.add_image("ants_image_random", img_random, i)

writer.close()