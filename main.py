import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
from numpy import linalg as LA

# generate the dataset
p = 0.8
q = 0.1
num_node = 100
C = 5
K = 1
block_size = (num_node / C * torch.ones(C,1)).squeeze().long()
edge_prob = q*torch.ones(C,C)
for i in range(C):
    edge_prob[i,i] = p

data_list = []
for i in range(2500):
    edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size,edge_prob)
    x = torch.randn(num_node,1)
    y = (sum(x.squeeze())/num_node * torch.ones(num_node,1))
    data = Data(x=x,y=y,edge_index=edge_index)

    data_list.append(data)
torch.save(data_list,'./data/data.pt')
print(data_list[0])

# split the data
from torch_geometric.data import DataLoader
data_list = torch.load('./data/data.pt')
train_data = data_list[:2000]
val_data = data_list[2000:2250]
test_data = data_list[2250:]

train_dataloader = DataLoader(train_data,batch_size=10,shuffle=True)

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
        for i in range(self.K):
#             print(self.weight.shape)
#
            if i == 0:
#                 print(input.shape)
#                 print(self.weight[i].shape)
                support = torch.mm(input, self.weight[i])
                out.append(support)
            else:
                tmp = torch.mm(adj,input)
                support = torch.mm(tmp, self.weight[i])
                out.append(support)
                adj = torch.mm(adj,adj)
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

        self.gc1 = GraphConvolution(K = K,in_features=nfeat, out_features=nhid)
        self.gc2 = GraphConvolution(K = K,in_features=nhid, out_features=nhid)
        self.lin = nn.Linear(nhid,nclasses)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.lin(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(nfeat=1,nhid=32,K = 2,nclasses=1)
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-2, weight_decay=5e-4)
crit = torch.nn.MSELoss()


# get the normalized adjacency matrix
from numpy import linalg as LA

def getNormalizedAdj(data):
    A = torch.zeros(data.x.shape[0],data.x.shape[0])
    A.to(device)
    for i in range(len(data.edge_index[1])//2):
        A[data.edge_index[0,2*i].numpy(),data.edge_index[0,2*i+1].numpy()] = 1
        A[data.edge_index[1,2*i].numpy(),data.edge_index[1,2*i+1].numpy()] = 1
    w, v = LA.eig(A)
    A = A / max(abs(w))
    return A

# train the model
import time

def train(epoch):
    t = time.time()
    model.train()
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()
        adj = getNormalizedAdj(data)
#         print(adj.shape)
        output = model(data.x, adj).squeeze()
        loss_train = crit(output,data.y.squeeze())
        loss_train.backward()
        optimizer.step()


    model.eval()
    loss_val = 0
    for i in range(10):
        val_data[i].to(device)
        adj = getNormalizedAdj(val_data[i])
        output = model(val_data[i].x, adj).squeeze()
        loss_val += crit(output,val_data[i].y.squeeze())
    loss_val = loss_val / 10

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

for epoch in range(1):
    train(epoch)
