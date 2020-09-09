import torch 
import torch.nn as nn 

class CosineLoss(nn.Module):
    def __init__(self, ):
        super(CosineLoss, self).__init__()
    

    def forward(self, x, y):
        return ((torch.cosine_similarity(x, y, dim=1)+1)).sum()

