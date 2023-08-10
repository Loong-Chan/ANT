import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math


class S2GC(nn.Module):
    def __init__(self, nfeat, nclass, features, adj, degree, alpha):
        super(S2GC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)
        self.feat = self.sgc_precompute(features, adj, degree, alpha)

    def forward(self, *args, **kwargs):
        return F.log_softmax(self.W(self.feat), dim=1)
    
    def sgc_precompute(self, features, adj, degree, alpha):
        emb = alpha * features
        for i in range(degree):
            features = torch.spmm(adj, features)
            emb = emb + (1-alpha)*features/degree
        return emb