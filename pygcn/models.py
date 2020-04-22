import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution1


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution1(nfeat, nhid)
        self.gc3 = GraphConvolution1(nhid, nhid)
        self.gc4 = GraphConvolution1(nhid, nhid)
        self.gc2 = GraphConvolution1(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

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
        self.gc4 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc5 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        self.gc2_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)

        self.gc3_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc4_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc5_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc6_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc7_1 = GraphConvolution(hidden_dim1+input_feat_dim, hidden_dim1, dropout, act=F.relu)

        self.node_regen = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)

        self.gc_class = GraphConvolution(hidden_dim1+input_feat_dim, nclass)

    def encode(self, x, adj, gc1, gc2, gc3, gc4, gc5):
        hidden1 = gc1(x, adj)
        return gc2(hidden1, adj), gc3(hidden1, adj), gc4(hidden1, adj), gc5(hidden1, adj), hidden1

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, mu_n, var_n, hidden1 = self.encode(x, adj, self.gc1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj1)
        masked_adj = torch.where(adj > 0, adj1, zero_vec)
        adj1 = F.softmax(masked_adj, dim=1)

        a1 = self.node_regen(z_n, adj1.t())
        zero_vec = -9e15*torch.ones_like(a1)
        masked_nodes = torch.where(x > 0, a1, zero_vec)
        a1 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  mu_n, var_n, hidden2 = self.encode(torch.cat([a1 , hidden1],-1), adj + adj1, self.gc2_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj2 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj2)
        masked_adj = torch.where(adj > 0, adj2, zero_vec)
        adj2 = F.softmax(masked_adj, dim=1)

        a2 = self.node_regen(z_n, adj2.t())
        zero_vec = -9e15*torch.ones_like(a2)
        masked_nodes = torch.where(x > 0, a2, zero_vec)
        a2 = F.softmax(masked_nodes, dim=1)


        mu, logvar,  mu_n, var_n, hidden3 = self.encode(torch.cat([a1+a2,hidden1 + hidden2],-1), adj + adj1 + adj2, self.gc3_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj3 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj3)
        masked_adj = torch.where(adj > 0, adj3, zero_vec)
        adj3 = F.softmax(masked_adj, dim=1)

        a3 = self.node_regen(z_n, adj3.t())
        zero_vec = -9e15*torch.ones_like(a3)
        masked_nodes = torch.where(x > 0, a3, zero_vec)
        a3 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  mu_n, var_n, hidden4 = self.encode(torch.cat([a1+a2 +a3,hidden1 + hidden2+hidden3],-1), adj + adj1 + adj2+adj3, self.gc4_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj4 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj4)
        masked_adj = torch.where(adj > 0, adj4, zero_vec)
        adj4 = F.softmax(masked_adj, dim=1)

        a4 = self.node_regen(z_n, adj4.t())
        zero_vec = -9e15*torch.ones_like(a4)
        masked_nodes = torch.where(x > 0, a4, zero_vec)
        a4 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  mu_n, var_n, hidden5 = self.encode(torch.cat([a1+a2 +a3 +a4,hidden1 + hidden2+hidden3+hidden4],-1), adj + adj1 + adj2+adj3+adj4, self.gc5_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj5 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj5)
        masked_adj = torch.where(adj > 0, adj5, zero_vec)
        adj5 = F.softmax(masked_adj, dim=1)

        a5 = self.node_regen(z_n, adj5.t())
        zero_vec = -9e15*torch.ones_like(a5)
        masked_nodes = torch.where(x > 0, a5, zero_vec)
        a5 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  mu_n, var_n, hidden6 = self.encode(torch.cat([a1+a2 +a3 +a4 +a5,hidden1 + hidden2+hidden3 + hidden4 + hidden5],-1), adj + adj1 + adj2+adj3+adj4+adj5, self.gc6_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj6 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj6)
        masked_adj = torch.where(adj > 0, adj6, zero_vec)
        adj6 = F.softmax(masked_adj, dim=1)

        a6 = self.node_regen(z_n, adj6.t())
        zero_vec = -9e15*torch.ones_like(a6)
        masked_nodes = torch.where(x > 0, a6, zero_vec)
        a6 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  mu_n, var_n, hidden7 = self.encode(torch.cat([a1+a2 +a3 +a4 +a5+a6,hidden1 + hidden2+hidden3 + hidden4 + hidden5+hidden6],-1), adj + adj1 + adj2+adj3+adj4+adj5+adj6, self.gc7_1, self.gc2, self.gc3, self.gc4, self.gc5)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj7 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj7)
        masked_adj = torch.where(adj > 0, adj7, zero_vec)
        adj7 = F.softmax(masked_adj, dim=1)

        a7 = self.node_regen(z_n, adj7.t())
        zero_vec = -9e15*torch.ones_like(a7)
        masked_nodes = torch.where(x > 0, a7, zero_vec)
        a7 = F.softmax(masked_nodes, dim=1)

        classifier = self.gc_class(torch.cat([a1+a2 +a3 +a4 +a5+a6 +a7,hidden1 + hidden2 + hidden3+ hidden4 + hidden5+hidden6 + hidden7],-1), adj + adj1 + adj2 + adj3+adj4+adj5+adj6 + adj7)

        return a1+a2+a3 + a4+a5+a6+a7, adj1 + adj2+ adj3+adj4+adj5+adj6 + adj7, mu, logvar, mu_n, var_n, F.log_softmax(classifier, dim=1)


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
