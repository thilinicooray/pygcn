from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import torch.optim as optim
import os
import glob

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.bemchmark=False

from pygcn.utils import load_data, accuracy
from pygcn.models import GCNModelVAE

def loss_function(preds, labels, mu, logvar, n_nodes):
    cost = F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# Load data
adj, adj1, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCNModelVAE(input_feat_dim=features.shape[1],
                    hidden_dim1=args.hidden,
                    hidden_dim2=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj1 = adj1.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


'''def train(epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()
    output = model(features, adj, adj1, fully_connected_graph)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, adj1, fully_connected_graph)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj, adj1, fully_connected_graph)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()'''


def train(epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()
    noderegen, recovered, mu, logvar, mu_n, var_n, output = model(features, adj1)
    node_cls_loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    ae_loss = loss_function(preds=recovered[idx_train], labels=adj1[idx_train],
                         mu=mu[idx_train], logvar=logvar[idx_train], n_nodes=features.shape[0])
    node_ae_loss = loss_function(preds=noderegen[idx_train], labels=features[idx_train],
                            mu=mu_n[idx_train], logvar=var_n[idx_train], n_nodes=features.shape[0])
    #print('losses ', node_cls_loss_train, ae_loss)
    loss_train = 2*node_cls_loss_train + 0.3*ae_loss + 0.3*node_ae_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    #if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
    model.eval()
    with torch.no_grad():
        noderegen, recovered, mu, logvar,mu_n, var_n, output = model(features, adj1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    '''print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))'''

    #return loss_val.data.item()
    return acc_val.data.item()


def compute_test():
    model.eval()
    with torch.no_grad():
        noderegen, recovered, mu, logvar, mu_n, var_n,output = model(features, adj1)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        loss_ae = loss_function(preds=recovered[idx_test], labels=adj1[idx_test],
                                mu=mu[idx_test], logvar=logvar[idx_test], n_nodes=features.shape[0])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "loss-AE= {:.4f}".format(loss_ae.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
acc_values = []
bad_counter = 0
best = 0
best_epoch = 0
for epoch in range(args.epochs):
    acc_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if acc_values[-1] > best:
        best = acc_values[-1]
        best_epoch = epoch
        bad_counter = 0
    '''else:
        bad_counter += 1

    if bad_counter == args.patience:
        break'''

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
