import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_edge


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc_e = GraphConvolution_edge(2*nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.emb = nn.Linear(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj, fully_connected_graph):
        x = self.emb(x)
        #making edge features
        conv1 = x.unsqueeze(1).expand(adj.size(0), adj.size(0), x.size(-1))
        conv2 = x.unsqueeze(0).expand(adj.size(0), adj.size(0), x.size(-1))
        conv1 = conv1.contiguous().view(-1, x.size(-1))
        conv2 = conv2.contiguous().view(-1, x.size(-1))

        edge_feat = torch.cat([conv1, conv2], -1)

        x_e = self.gc_e(edge_feat, adj)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.joint(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
