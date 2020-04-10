import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.joint = nn.Linear(nfeat + nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, fully_connected_graph):
        x_init = x
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.joint(torch.cat([x_init, x], -1))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
