import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.nnode = nnode
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1[edge_index[0]] + Wh2[edge_index[1]]).reshape(-1)
        e = torch.sparse_coo_tensor(edge_index, e, torch.Size([self.nnode, self.nnode]))
        e = torch.sparse.softmax(e, dim=1)
        h_prime = torch.sparse.mm(e, Wh).to_dense()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nnode, nfeat, nhid, nclass, dropout, nheads, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nnode, nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nnode, nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.log_softmax(x, dim=1)