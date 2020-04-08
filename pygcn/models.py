import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.embedding_h = nn.Linear(nfeat, nhid)
        self.joint = nn.Linear(nhid, nhid)

    def forward(self, x, adj):
        x_init = self.embedding_h(x)
        x = F.relu(self.gc1(x_init, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(F.tanh(self.joint(x + x_init)))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
