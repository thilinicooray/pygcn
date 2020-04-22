'''import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, adj1, fc):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygcn.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, nclass, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc1 = InnerProductDecoder(dropout, act=lambda x: x)

        self.gc2_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc2_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc2_3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.gc3_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc3_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.gc_class = GraphConvolution(hidden_dim1, nclass)

    def encode(self, x, adj, gc1, gc2, gc3):
        hidden1 = gc1(x, adj)
        return gc2(hidden1, adj), gc3(hidden1, adj), hidden1

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, hidden1 = self.encode(x, adj, self.gc1, self.gc2, self.gc3)
        z = self.reparameterize(mu, logvar)
        pred_a1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(pred_a1)
        masked_adj = torch.where(adj > 0, pred_a1, zero_vec)
        new_adj = F.softmax(masked_adj, dim=1)

        mu, logvar, hidden2 = self.encode(hidden1, new_adj, self.gc2_1, self.gc2, self.gc3)
        z = self.reparameterize(mu, logvar)
        pred_a1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(pred_a1)
        masked_adj = torch.where(adj > 0, pred_a1, zero_vec)
        new_adj = F.softmax(masked_adj, dim=1)

        mu, logvar, hidden3 = self.encode(hidden2, new_adj, self.gc3_1, self.gc2, self.gc3)
        z = self.reparameterize(mu, logvar)
        pred_a1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(pred_a1)
        masked_adj = torch.where(adj > 0, pred_a1, zero_vec)
        new_adj = F.softmax(masked_adj, dim=1)



        classifier = self.gc_class(hidden3, new_adj)

        return new_adj, mu, logvar, F.log_softmax(classifier, dim=1)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
