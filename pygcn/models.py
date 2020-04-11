import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_edge


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc_e = GraphConvolution_edge(nhid, nhid)
        self.gc_e2 = GraphConvolution_edge(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.emb = nn.Linear(nfeat, nhid)
        self.cls = nn.Linear(nhid, nclass)
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
        x1 = F.dropout(x, self.dropout, training=self.training)
        #x = self.joint(x)
        x = x1 + x_e + x_init
        x = self.gc2(x, adj)

        conv1 = x1.unsqueeze(1).expand(adj.size(0), adj.size(0), x1.size(-1))
        conv2 = x1.unsqueeze(0).expand(adj.size(0), adj.size(0), x1.size(-1))
        conv1 = conv1.contiguous().view(-1, x1.size(-1))
        conv2 = conv2.contiguous().view(-1, x1.size(-1))

        #edge_feat = torch.cat([conv1, conv2], -1)
        edge_feat1 = conv1 * conv2
        x_e = edge_feat + edge_feat1
        x_e = self.gc_e2(x_e, adj1)

        x = F.dropout(F.relu(x + x_e), self.dropout, training=self.training)

        x = self.cls(x)


        return F.log_softmax(x, dim=1)
