import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
import torch_geometric
import time
import os
import argparse
# device = 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=100,
                    help='number of the node')
parser.add_argument('--K', type=int, default=25,
                    help='the size of the filter order')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the size of a batch')
args = parser.parse_args()


os.makedirs('./same_data_diffirent_try', exist_ok=True)
os.chdir('./same_data_diffirent_try')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#define the layer
class GraphConvolution(Module):
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
            if i == 0:
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

#define the model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, K, nclasses):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(K=K, in_features=nfeat, out_features=nhid)
        self.gc2 = GraphConvolution(K=K, in_features=nhid, out_features=nhid)
        self.bn1 = torch.nn.BatchNorm1d(nhid)
        self.bn2 = torch.nn.BatchNorm1d(nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, adj):
        x = self.bn1(self.gc1(x, adj))
        x = F.relu(x)
        x = self.bn2(self.gc2(x, adj))
        x = F.relu(x)
        x = self.lin(x)
        return x
#get the normalized adjacency
def getNormalizedAdj(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()
    evals_large, evecs_large = eigsh(adj, 1, which='LM')
    adj = torch.Tensor(adj / evals_large)
    adj = adj.to(device)
    return adj

def getTrainNormalizedAdj(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()
    for d in range(args.batch_size):
        index1 = d*args.num_node
        index2 = (d+1)*args.num_node
        # print(index1,index2)
        evals_large, evecs_large = eigsh(adj[index1:index2,index1:index2],
                                         1, which='LM')
        adj[index1:index2, index1:index2] = adj[index1:index2,index1:index2] / evals_large

    adj = torch.Tensor(adj)
    adj = adj.to(device)
    return adj

#dataset generation
from torch_geometric.data import Data
#
# p = 0.8
# q = 0.1
# C = 5
# block_size = (args.num_node / C * torch.ones(C, 1)).squeeze().long()
#
#
# edge_prob = q * torch.ones(C, C)
# for i in range(C):
#     edge_prob[i, i] = p
#
# data_list = []
# val_data_list = []
# test_data_list = []
# for i in range(args.data_size):
#     edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size, edge_prob,directed=False)
#
#     x = torch.randn(args.num_node, 1)
#
#     y = (sum(x.squeeze()) / args.num_node * torch.ones(args.num_node))
#     data = Data(x=x, y=y, edge_index=edge_index)
#
#     data_list.append(data)
# for j in range(2):
#     for i in range(2500):
#         edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size, edge_prob,directed=False)
#         x = torch.randn(args.num_node, 1)
#         y = (sum(x.squeeze()) / args.num_node * torch.ones(args.num_node))
#         data = Data(x=x, y=y, edge_index=edge_index)
#         if j == 0 :
#             val_data_list.append(data)
#         else:
#             test_data_list.append(data)
#
# train_data = data_list[:int(0.8 * args.data_size)]
# val_data = val_data_list
# test_data = test_data_list
data_list = torch.load('./data_list.pt')
val_data_list = torch.load('./val_data_list.pt')
test_data_list = torch.load('./test_data_list.pt')

train_data = data_list[:int(0.8 * args.data_size)]
val_data = val_data_list
test_data = test_data_list
print(len(test_data),len(val_data),len(train_data))
os.makedirs('./batchsize_%d_Nov7th_revise'%(args.batch_size), exist_ok=True)
os.chdir('./batchsize_%d_Nov7th_revise'%(args.batch_size))

from torch_geometric.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

model = GCN(nfeat=1, nhid=32, K=args.K+1, nclasses=1)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
crit = torch.nn.MSELoss()

fo = open("output.txt","w+")
model.train()
ll_train = 1e6
ll_val = 1e6
ll_val_lowest = 1e7
k = 0
cur_time = time.time()
# while(ll_val>1e-4):
while(k<1200):
    k += 1
    loss_train_list = []
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()
        # adj = getTrainNormalizedAdj(data)
        adj = getNormalizedAdj(data)
        output = model(data.x, adj).squeeze()
        loss_train = crit(output, data.y.squeeze())

        loss_train_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()
    ll_train = sum(loss_train_list) / len(train_data)
    print(ll_train)
    fo.write('Epoch: {:04d},'.format(k))
    fo.write('loss_train: {:.11f},'.format(ll_train))

    loss_val_list = []
    for j in range(len(val_data)):
        data = val_data[j]
        data.to(device)
        adj = getNormalizedAdj(data)
        output = model(data.x, adj).squeeze()
        loss_val = crit(output, data.y.squeeze())

        loss_val_list.append(loss_val.item())

    ll_val = sum(loss_val_list) / len(val_data)
    print(ll_val)
    if ll_val < ll_val_lowest:
        ll_val_lowest = ll_val
        torch.save(model, './model_best.pt')
    fo.write('loss_val: {:.11f}\n'.format(ll_val))

    if k % 1000 == 0:
        torch.save(model, './model_epoch_%d.pt'%(k%1000))

torch.save(model, './model.pt')

end_time = time.time()
# model.eval()
loss_test_list = []
for j in range(len(test_data)):
    data = test_data[j]
    data.to(device)
    adj = getNormalizedAdj(data)
    output = model(data.x, adj).squeeze()
    loss_test = crit(output, data.y.squeeze())
    print(output,"\n",data.y.squeeze())
    print(loss_test.item())
    loss_test_list.append(loss_test.item())
    fo.write('data: {:04d},'.format(j))
    fo.write('loss_test: {:.11f}\n'.format(loss_test.item()))
print("oooooooooooooooooooooooooooooooooooooooooooooooo")
print('average loss: %f'%(sum(loss_test_list)/len(test_data)))
print(k)
print(end_time-cur_time,"s")
fo.write('average loss: %f,'%(sum(loss_test_list)/len(test_data)))
fo.write('the number of epoch:%d,'%(k))
fo.write('total time:%f'%(end_time-cur_time))
fo.write('s\n')
