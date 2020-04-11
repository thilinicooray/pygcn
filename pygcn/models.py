import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_edge


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc_e = GraphConvolution_edge(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.emb = nn.Linear(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj, adj1, fully_connected_graph):
        x_init = self.emb(x)
        #making edge features
        conv1 = x_init.unsqueeze(1).expand(adj.size(0), adj.size(0), x_init.size(-1))
        conv2 = x_init.unsqueeze(0).expand(adj.size(0), adj.size(0), x_init.size(-1))
        conv1 = conv1.contiguous().view(-1, x_init.size(-1))
        conv2 = conv2.contiguous().view(-1, x_init.size(-1))

        #edge_feat = torch.cat([conv1, conv2], -1)
        edge_feat = conv1 * conv2

        x_e = F.relu(self.gc_e(edge_feat, adj1))
        x_e = F.dropout(x_e, self.dropout, training=self.training)
        x = F.relu(self.gc1(x_init, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.joint(x)
        x = x + x_e + x_init
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
