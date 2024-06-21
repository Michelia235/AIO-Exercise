#Câu 1 tự luận:
import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

class softmax_stable(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        partition = x_exp.sum(0, keepdim=True)
        return x_exp / partition 

#ví dụ trong slide câu 1.1:
data = torch.Tensor([1,2,3])
softmax = Softmax()
output = softmax(data)  
print(output)