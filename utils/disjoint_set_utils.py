
import torch
import torch.nn as nn

class DisjointSet(nn.Module):
    def __init__(self, number_points:int):
        super().__init__()

        self.father = torch.arange(number_points).to(torch.int64)
        if torch.cuda.is_available():
            self.father = self.father.cuda()

    def find(self, x):
        if x == self.father[x]:
            return x
        self.father[x] = self.find(self.father[x])

        return self.father[x]

    def union(self, a, b):
        af = self.find(a)
        bf = self.find(b)

        if af < bf:
            self.father[bf] = af
        else:
            self.father[af] = bf

    def densify(self):
        last_father = self.father.clone()
        self.father = self.father[self.father].clone()
        while torch.any(torch.ne(last_father, self.father)).item():
            last_father = self.father.clone()
            self.father = self.father[self.father].clone()

    def get_father_with_mask(self, mask):
        assert mask.shape[0] == self.father.shape[0]

        return self.father[mask]