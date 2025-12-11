import torch

import numpy as np
import torch

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

a = torch.Tensor([[1, 2],[3, 4]])
print(a)

b = torch.randn(2,3)
print(b)

c = torch.normal(mean=0.0, std=torch.rand(3,4))
print("c:{}".format(c))

### 序列
d = torch.arange(1,11,1)
print("d:{}".format(d))

i = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float32,device=torch.device('cpu'))
print("i:{} ".format(i))

print(torch.cuda.is_available())