import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_edge

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network with gated tangent as in paper
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        in_dim = dims[0]
        out_dim = dims[1]
        self.first_lin = nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()
        self.second_lin = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        y_hat = self.tanh(self.first_lin(x))
        g = self.sigmoid(self.second_lin(x))
        y = y_hat * g

        return y


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nclass)

        self.confidence = nn.Sequential(nn.Linear(nhid*2, nhid),

                                        nn.Linear(nhid, 1),
                                        
                                        )

        self.confidence2 = nn.Sequential(nn.Linear(nhid*2, nhid),
                                        nn.ReLU(),
                                        nn.Linear(nhid, 1),
                                        nn.Sigmoid())
        self.dropout = dropout

    def forward1(self, x, adj, adj1, fully_connected_graph):
        '''x_init = self.emb(x)
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

        #x = F.dropout(F.relu(x + x_e), self.dropout, training=self.training)
        x = x + x_e
        x = self.cls(x)


        return F.log_softmax(x, dim=1)'''

    def forward(self, x_init, adj, adj1_org, fully_connected_graph):

        x = torch.tanh(self.gc1(x_init, adj1_org))
        x = F.dropout(x, self.dropout, training=self.training)

        #if self.training:

        conv1 = x.unsqueeze(1).expand(adj.size(0), adj.size(0), x.size(-1))
        conv2 = x.unsqueeze(0).expand(adj.size(0), adj.size(0), x.size(-1))
        #conv1 = conv1.contiguous().view(-1, x.size(-1))
        #conv2 = conv2.contiguous().view(-1, x.size(-1))

        edge_feat = torch.cat([conv1, conv2], -1)

        edge_feat = self.confidence(edge_feat)
        print('edge_feat ', edge_feat[0,0,:10], adj1_org[0,:10])


        scores = edge_feat.masked_fill(edge_feat > 0, 1).squeeze()
        adj1 = adj1_org * scores

        #adj1 = adj1 + adj1_org

        x = torch.cat([x, x_init], -1)

        x = self.gc2(x, adj1)

        return F.log_softmax(x, dim=1)

