import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.JOINT_EMB_SIZE = 3 * out_features
        self.Linear_nodeproj = nn.Linear(in_features, self.JOINT_EMB_SIZE)
        self.Linear_neighbourproj = nn.Linear(in_features, self.JOINT_EMB_SIZE)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        data_out = self.Linear_dataproj(input)                   # data_out (batch, 5000)
        img_feature = self.Linear_imgproj(input)      # img_feature (batch, 5000)
        iq = torch.mul(data_out, img_feature)
        iq = F.dropout(iq, 0.1, training=self.training)
        iq = iq.view(-1, 1, self.out_features, 3)
        iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
        iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))       # signed sqrt
        support = F.normalize(iq)


        print('support size ', support.size(), input.size(), adj.size())

        output = torch.spmm(adj, support)
        print('output ', output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
