import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution1

import numpy as np
import scipy.sparse as sp


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

        self.gc2_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)

        self.gc3_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc4_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc5_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc6_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc7_1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)


        self.gc2_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc2_3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc2_4 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_4 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc2_5 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_5 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc2_6 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3_6 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.node_regen = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)

        self.adj2node = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(1, hidden_dim1),
            nn.ReLU(inplace=True),
        )

        self.gc_class = GraphConvolution(hidden_dim1, nclass)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        '''mx = mx.cpu().detach().numpy()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)'''

        mx = F.normalize(mx, p=2, dim=1)
        return mx

    def encode(self, x, adj, gc1, gc2, gc3):
        hidden1 = gc1(x, adj)
        return gc2(hidden1, adj), gc3(hidden1, adj),  hidden1

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward1(self, x, adj):
        #print('layer init adj ', adj[:2,:10])
        #print('layer init node ', x[:2,:10])

        mu, logvar, hidden1 = self.encode(x, adj, self.gc1, self.gc2, self.gc3)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj1)
        masked_adj = torch.where(adj > 0, adj1, zero_vec)
        adj1 = F.softmax(masked_adj, dim=1)


        a1 = self.node_regen(z, adj1.t())
        zero_vec = -9e15*torch.ones_like(a1)
        masked_nodes = torch.where(x > 0, a1, zero_vec)
        a1 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  hidden2 = self.encode(torch.cat([a1 , hidden1 ],-1), adj + adj1, self.gc2_1, self.gc2_2, self.gc3_2)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj2 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj2)
        masked_adj = torch.where(adj > 0, adj2, zero_vec)
        adj2 = F.softmax(masked_adj, dim=1)


        a2 = self.node_regen(z, adj2.t())
        zero_vec = -9e15*torch.ones_like(a2)
        masked_nodes = torch.where(x > 0, a2, zero_vec)
        a2 = F.softmax(masked_nodes, dim=1)


        mu, logvar,   hidden3 = self.encode(torch.cat([a2,hidden1 + hidden2 ],-1), adj + adj1 + adj2, self.gc3_1, self.gc2_3, self.gc3_3)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj3 = self.dc(z)

        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj3)
        masked_adj = torch.where(adj > 0, adj3, zero_vec)
        adj3 = F.softmax(masked_adj, dim=1)

        #print('layer 3 adj ', adj3[:2,:10])


        a3 = self.node_regen(z, adj3.t())
        zero_vec = -9e15*torch.ones_like(a3)
        masked_nodes = torch.where(x > 0, a3, zero_vec)
        a3 = F.softmax(masked_nodes, dim=1)
        #print('layer 3 nodes ', a3[:2,:10])

        mu, logvar,  hidden4 = self.encode(torch.cat([a3,hidden1 + hidden2+hidden3],-1), adj + adj1 + adj2+adj3, self.gc4_1, self.gc2_4, self.gc3_4)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj4 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj4)
        masked_adj = torch.where(adj > 0, adj4, zero_vec)
        adj4 = F.softmax(masked_adj, dim=1)

        a4 = self.node_regen(z, adj4.t())
        zero_vec = -9e15*torch.ones_like(a4)
        masked_nodes = torch.where(x > 0, a4, zero_vec)
        a4 = F.softmax(masked_nodes, dim=1)

        mu, logvar,  hidden5 = self.encode(torch.cat([a4,hidden1 + hidden2+hidden3+hidden4],-1), adj + adj1 + adj2+adj3+adj4, self.gc5_1, self.gc2_5, self.gc3_5)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj5 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj5)
        masked_adj = torch.where(adj > 0, adj5, zero_vec)
        adj5 = F.softmax(masked_adj, dim=1)

        #print('layer 5 adj ', adj5[:2,:10])

        a5 = self.node_regen(z, adj5.t())
        zero_vec = -9e15*torch.ones_like(a5)
        masked_nodes = torch.where(x > 0, a5, zero_vec)
        a5 = F.softmax(masked_nodes, dim=1)
        #print('layer 5 nodes ', a5[:2,:10])

        mu, logvar, hidden6 = self.encode(torch.cat([a5,hidden1 + hidden2+hidden3+hidden4+hidden5],-1), adj + adj1 + adj2+adj3+adj4+adj5, self.gc6_1, self.gc2_6, self.gc3_6)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj6 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj6)
        masked_adj = torch.where(adj > 0, adj6, zero_vec)
        adj6 = F.softmax(masked_adj, dim=1)
        #print('layer 6 adj ', adj6[:2,:10])

        a6 = self.node_regen(z, adj6.t())
        zero_vec = -9e15*torch.ones_like(a6)
        masked_nodes = torch.where(x > 0, a6, zero_vec)
        a6 = F.softmax(masked_nodes, dim=1)
        #print('layer 6 nodes ', a6[:2,:10])

        '''mu, logvar,  mu_n, var_n, hidden7 = self.encode(torch.cat([a6,hidden1 + hidden2+hidden3+hidden4+hidden5+hidden6],-1), adj + adj1 + adj2+adj3+adj4+adj5+adj6, self.gc7_1, self.gc2, self.gc3, self.gc4, self.gc5)
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


        classifier = self.gc_class(torch.cat([a7,hidden1 + hidden2 + hidden3 + hidden4 + hidden5 + hidden6 + hidden7 ],-1), adj + adj1 + adj2 + adj3 + adj4+adj5+adj6 + adj7)

        return a1+a2+a3+ a4+ a5+ a6+ a7, adj1 + adj2+ adj3+ adj4+adj5+adj6 + adj7, mu, logvar, mu_n, var_n, F.log_softmax(classifier, dim=1)'''

        classifier = self.gc_class(torch.cat([a6,hidden1 + hidden2 + hidden3+hidden4 + hidden5+hidden6  ],-1), adj + adj1 + adj2 + adj3+adj4+adj5+adj6)

        return a1+a2+a3+a4+a5+a6, adj1 + adj2+ adj3 + adj4+adj5+adj6, mu, logvar, F.log_softmax(classifier, dim=1)

    def forward(self, x, adj):
        #print('layer init adj ', adj[:2,:10])
        #print('layer init node ', x[:2,:10])

        mu, logvar, hidden1 = self.encode(x, adj, self.gc1, self.gc2, self.gc3)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj1)
        masked_adj = torch.where(adj > 0, adj1, zero_vec)
        adj1 = F.softmax(masked_adj, dim=1)




        mu, logvar,  hidden2 = self.encode(hidden1, adj + adj1, self.gc2_1, self.gc2_2, self.gc3_2)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj2 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj2)
        masked_adj = torch.where(adj > 0, adj2, zero_vec)
        adj2 = F.softmax(masked_adj, dim=1)





        mu, logvar,   hidden3 = self.encode(hidden1 + hidden2, adj + adj1 + adj2, self.gc3_1, self.gc2_3, self.gc3_3)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj3 = self.dc(z)

        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj3)
        masked_adj = torch.where(adj > 0, adj3, zero_vec)
        adj3 = F.softmax(masked_adj, dim=1)

        #print('layer 3 adj ', adj3[:2,:10])


        mu, logvar,  hidden4 = self.encode(hidden1 + hidden2+hidden3, adj + adj1 + adj2+adj3, self.gc4_1, self.gc2_4, self.gc3_4)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj4 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj4)
        masked_adj = torch.where(adj > 0, adj4, zero_vec)
        adj4 = F.softmax(masked_adj, dim=1)

        mu, logvar,  hidden5 = self.encode(hidden1 + hidden2+hidden3+hidden4, adj + adj1 + adj2+adj3+adj4, self.gc5_1, self.gc2_5, self.gc3_5)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj5 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj5)
        masked_adj = torch.where(adj > 0, adj5, zero_vec)
        adj5 = F.softmax(masked_adj, dim=1)

        #print('layer 5 adj ', adj5[:2,:10])



        mu, logvar, hidden6 = self.encode(hidden1 + hidden2+hidden3+hidden4+hidden5, adj + adj1 + adj2+adj3+adj4+adj5, self.gc6_1, self.gc2_6, self.gc3_6)
        z = self.reparameterize(mu, logvar)
        #z_n = self.reparameterize(mu_n, var_n)
        adj6 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj6)
        masked_adj = torch.where(adj > 0, adj6, zero_vec)
        adj6 = F.softmax(masked_adj, dim=1)
        #print('layer 6 adj ', adj6[:2,:10])



        '''mu, logvar,  mu_n, var_n, hidden7 = self.encode(torch.cat([a6,hidden1 + hidden2+hidden3+hidden4+hidden5+hidden6],-1), adj + adj1 + adj2+adj3+adj4+adj5+adj6, self.gc7_1, self.gc2, self.gc3, self.gc4, self.gc5)
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


        classifier = self.gc_class(torch.cat([a7,hidden1 + hidden2 + hidden3 + hidden4 + hidden5 + hidden6 + hidden7 ],-1), adj + adj1 + adj2 + adj3 + adj4+adj5+adj6 + adj7)

        return a1+a2+a3+ a4+ a5+ a6+ a7, adj1 + adj2+ adj3+ adj4+adj5+adj6 + adj7, mu, logvar, mu_n, var_n, F.log_softmax(classifier, dim=1)'''

        classifier = self.gc_class(hidden1 + hidden2 + hidden3+hidden4 + hidden5+hidden6 , adj + adj1 + adj2 + adj3+adj4+adj5+adj6)

        return  adj1 + adj2+ adj3 + adj4+adj5+adj6, mu, logvar, F.log_softmax(classifier, dim=1)


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
