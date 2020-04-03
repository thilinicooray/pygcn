import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nfeat, nhid)
        self.combiner = nn.Linear(nhid + nhid, nhid)
        self.dropout = dropout

    def forward(self, x_org, adj):
        x_org = self.encoder(x_org)
        x = F.relu(self.gc1(x_org, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        x = x1 + x_org
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc2(x, adj)
        x2 = F.dropout(x, self.dropout, training=self.training)
        x = x1 + x2
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)
