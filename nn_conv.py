import torch


input = torch.tensor([[1, 3, 32, 32],[2, 3, 32, 32],[3, 3, 32, 32]])

kernel = torch.tensor([[1, 3, 2],[1, 2, 0],[1, 1, 0]])

input = torch.reshape(input, (1,1,4,3))
kernel = torch.reshape(kernel, (1,1,3,3))

print(input.shape)
print(kernel.shape)

output2 = torch.nn.functional.conv2d(input, kernel, stride=1)
print(output2)
output3 = torch.nn.functional.conv2d(input, kernel, stride=1, padding=1)
print("output3:{}".format(output3))