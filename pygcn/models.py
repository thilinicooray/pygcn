import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc1_1 = GraphConvolution(nfeat, nhid)
        self.gc2_1 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc1_1(x, adj))
        x = x1 * x2
        x = F.dropout(x, self.dropout, training=self.training)
        x3 = self.gc2(x, adj)
        x4 = self.gc2_1(x, adj)
        x = x3 * x4
        return F.log_softmax(x, dim=1)
