import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import math


class BiasedModel(nn.Module):
    def __init__(self, C, I, K = 10, bias = True):
        super().__init__()
        self.K = K
        self.I = I
        self.fc = nn.Linear(C + I, K, bias=bias)
        # self.fc_item = nn.Linear(I, 32)
        # self.fc2 = nn.Linear(32, 32)
        # self.fc3 = nn.Linear(32, K)
        nn.init.kaiming_uniform_(self.fc.weight)
        # self.fc_v = nn.Linear(K, I, bias= False)
        # nn.init.uniform_(self.fc_v.weight)
        # self.relu = nn.ReLU()
        self.common_uk = nn.parameter.Parameter(torch.empty((1, K)))
        self.v_c = nn.parameter.Parameter(torch.empty((K, C)))
        self.v_i = nn.parameter.Parameter(torch.empty((K, I)))
        nn.init.kaiming_uniform_(self.common_uk)
        nn.init.kaiming_uniform_(self.v_c)
        nn.init.kaiming_uniform_(self.v_i)
        # self.v.relu_()
    def forward(self, cate, item):
        # print(torch.cat((cate, item).shape))
        x = self.fc(torch.cat((cate, item), dim = 1))
        # print(x.shape)
        x = F.softmax(x)
        # x = self.fc2(x)
        # x=  F.relu(x)
        # x = self.fc3(x)
        # ones_ = torch.ones((self.K, self.K))
        common_uk_ = F.softmax(self.common_uk)
        # print(x.shape)

        u = torch.add(x, common_uk_)
        u = F.softmax(u)
        # v_ = self.fc_v(ones_)
        v_c_ = F.softmax(self.v_c)
        v_i_ = F.softmax(self.v_i)
        # self.v.data = v_
        # r = self.fc_v(x)
        # with torch.no_grad():
            # v_ = self.fc_v.weight.detach()
        # print()
        # print(v_)
        # print(self.v.grad)

        return x, common_uk_, u, v_c_, v_i_, torch.matmul(u, v_c_), torch.matmul(u, v_i_)
        # return x, v_, 

class CustomDataset(Dataset):
    def __init__(self, x_cate, x_item):
        self.x_cate = x_cate
        self.x_item = x_item
        
    def __getitem__(self, index):
        return (self.x_cate[index, :], self.x_item[index, :])
    
    def __len__(self):
        return self.x_cate.size(1)

