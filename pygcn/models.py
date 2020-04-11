import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc_e = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.joint = nn.Linear(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, fully_connected_graph):
        #making edge features
        conv1 = x.unsqueeze(1).expand(adj.size[0], adj.size[0], x.size(-1))
        conv2 = x.unsqueeze(0).expand(adj.size[0], adj.size[0], x.size(-1))
        conv1 = conv1.contiguous().view(-1, x.size(-1))
        conv2 = conv2.contiguous().view(-1, x.size(-1))

        print(conv1[:5], conv2[:5])


        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.joint(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
