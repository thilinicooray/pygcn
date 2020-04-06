import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.nhid = nhid
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nfeat, nhid)
        self.combiner = nn.Linear(nhid + nhid, nhid)
        self.dropout = dropout

        self.JOINT_EMB_SIZE = 5 * nhid
        self.Linear_nodeproj = nn.Linear(nhid, self.JOINT_EMB_SIZE)
        self.Linear_neighourproj = nn.Linear(nhid, self.JOINT_EMB_SIZE)

    def forward(self, x_org, adj):
        '''x_org = self.encoder(x_org)
        x = F.relu(self.gc1(x_org, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        x = x1 + x_org
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc2(x, adj)
        x2 = F.dropout(x, self.dropout, training=self.training)
        x = x1 + x2
        #x = self.combiner(torch.cat([x, x_org], -1))
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)'''

        x_org = self.encoder(x_org)
        x = self.gc1(x_org, adj)

        node_out = self.Linear_nodeproj(x_org)                   # data_out (batch, 5000)
        neighbour_out = self.Linear_neighourproj(x)      # img_feature (batch, 5000)
        iq = torch.mul(node_out, neighbour_out)
        iq = F.dropout(iq, self.dropout, training=self.training)
        iq = iq.view(-1, 1, self.nhid, 5)
        iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
        iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))       # signed sqrt
        x1 = F.normalize(iq)

        x = self.gc2(x1, adj)
        node_out = self.Linear_nodeproj(x1)                   # data_out (batch, 5000)
        neighbour_out = self.Linear_neighourproj(x)      # img_feature (batch, 5000)
        iq = torch.mul(node_out, neighbour_out)
        iq = F.dropout(iq, self.dropout, training=self.training)
        iq = iq.view(-1, 1, self.nhid, 5)
        iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
        iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))       # signed sqrt
        x2 = F.normalize(iq)

        x = self.gc3(x2, adj)

        return F.log_softmax(x, dim=1)
