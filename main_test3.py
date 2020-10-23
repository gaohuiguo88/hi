import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
from numpy import linalg as LA
import argparse
import warnings
import os

from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train')
parser.add_argument('--K', type=int, default=25,
                    help='Number of filter order')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Size of the batch')
parser.add_argument('--q', type=float, default=0.1,
                    help='inter-community probabilities')
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of generated data')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

args = parser.parse_args()
os.makedirs('./Relu_K_%d_lr_%f' % (args.K, args.lr), exist_ok=True)
os.chdir('./Relu_K_%d_lr_%f' % (args.K, args.lr))

# generate the dataset
p = 0.8
q = args.q
num_node = 100
C = 5
block_size = (num_node / C * torch.ones(C, 1)).squeeze().long()
edge_prob = q * torch.ones(C, C)
for i in range(C):
    edge_prob[i, i] = p

data_list = []
for i in tqdm(range(args.data_size)):
    edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size, edge_prob)
    x = torch.randn(num_node, 1)
    y = (sum(x.squeeze()) / num_node * torch.ones(num_node))
    data = Data(x=x, y=y, edge_index=edge_index)

    data_list.append(data)
torch.save(data_list, './data.pt')
print(data_list[0])

# split the data
from torch_geometric.data import DataLoader

data_list = torch.load('./data.pt')
train_data = data_list[:int(0.8 * args.data_size)]
val_data = data_list[int(0.8 * args.data_size):int(0.9 * args.data_size)]
test_data = data_list[int(0.9 * args.data_size):]

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# define the layer
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, K, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        out = []
        adj_ = adj
        for i in range(self.K):
            #             print(self.weight.shape)
            #
            if i == 0:
                #                 print(input.shape)
                #                 print(self.weight[i].shape)
                support = torch.mm(input, self.weight[i])
                out.append(support)
            else:
                tmp = torch.mm(adj_, input)
                support = torch.mm(tmp, self.weight[i])
                out.append(support)
                adj_ = torch.mm(adj_, adj)

        output = sum(out)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# define the model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, K, nclasses):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(K=K, in_features=nfeat, out_features=nhid)
        self.gc2 = GraphConvolution(K=K, in_features=nhid, out_features=nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.lin(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = GCN(nfeat=1, nhid=32, K=args.K, nclasses=1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
crit = torch.nn.MSELoss()

# get the normalized adjacency matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def getNormalizedAdj(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    #    raw_data = raw_data.to(device)

    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()

    evals_large, evecs_large = eigsh(adj, 1, which='LM')
    adj = torch.Tensor(adj / evals_large)

    adj = adj.to(device)

    return adj


# train the model
loss_val_best = 1e9

import time


def train(epoch):
    t = time.time()
    global loss_val_best
    loss_train_list = []
    model.train()
    for data in tqdm(train_dataloader):
        data.to(device)
        optimizer.zero_grad()
        adj = getNormalizedAdj(data)
        output = model(data.x, adj).squeeze()
        loss_train = crit(output, data.y.squeeze())
        loss_train_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

    model.eval()
    loss_val = 0
    for i in range(250):
        val_data[i].to(device)
        adj = getNormalizedAdj(val_data[i])
        output = model(val_data[i].x, adj).squeeze()
        loss_val += crit(output, val_data[i].y.squeeze())
    loss_val = loss_val / 250
    if loss_val < loss_val_best:
        torch.save(model, './best_model.pt')
        loss_val_best = loss_val

    loss_retrain = 0
    for i in range(250):
        train_data[i].to(device)
        adj = getNormalizedAdj(train_data[i])
        output = model(train_data[i].x, adj).squeeze()
        loss_retrain += crit(output, train_data[i].y.squeeze())
        train_data[i].to('cpu')
    loss_retrain = loss_retrain / 250

    fo.write('Epoch: {:04d},'.format(epoch + 1))
    fo.write('loss_train: {:.11f},'.format(sum(loss_train_list) / len(loss_train_list)))
    fo.write('loss_val: {:.11f},'.format(loss_val.item()))
    fo.write('loss_retrain: {:.11f},'.format(loss_retrain.item()))
    fo.write('time: {:.4f}s\n'.format(time.time() - t))


fo = open('output.txt', 'w+')

for epoch in range(args.epochs):
    train(epoch)
    torch.save(model, './model_epoch_%d.pt' % epoch)
fo.close()
torch.cuda.empty_cache()