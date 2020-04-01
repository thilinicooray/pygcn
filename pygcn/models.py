import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.encoder1 = nn.Parameter(torch.zeros(size=(nfeat, nhid)))
        self.encoder = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=False)
        self.combiner = nn.Parameter(torch.zeros(size=(nhid + nhid, nhid)))
        self.dropout = dropout

    def forward(self, x_org, adj):
        x_proj = self.encoder1(x_org)
        x_entire = self.encoder(x_proj)
        x_entire = x_proj * x_entire
        x = F.relu(self.gc1(x_entire, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)
