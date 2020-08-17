__doc__ = """Tree GRU aka Recursive Neural Networks."""
# coding=utf-8
import numpy as np

#from collections import OrderedDict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# lstm 模型架构
class Node_tweet(object):
    def __init__(self, idx=None):
        super(Node_tweet, self).__init__()
        self.children = []
        #self.index = index
        self.idx = idx
        self.word = []
        self.index = []
        self.numchildren = 0
        #self.height = 1
        #self.size = 1
        #self.num_leaves = 1
        self.parent = None
        #self.label = None
# 模型架构
#gru sum模型架构
class ChildSumTreeGRU(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, Nclass):
        super(ChildSumTreeGRU, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.Embedding = nn.Embedding(vocab_size, in_dim)
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.wx, self.wh, self.b = self.get_params(in_dim, mem_dim)
        self.fc = nn.Linear(self.mem_dim, Nclass)
    def get_params(self, in_dim, mem_dim): # 获取参数
        def _one(shape):
            ts = torch.tensor(np.random.normal(0, 0.1, size=shape), device=device,dtype=torch.float32)
            return nn.Parameter(ts, requires_grad=True)
        return (
            _one((in_dim, mem_dim)),
            _one((mem_dim, mem_dim)),
            nn.Parameter(torch.zeros(mem_dim, device=device, dtype=torch.float32), requires_grad=True)
        )
    def node_forward(self, tree, child_h):
        child_h_Sum = torch.sum(child_h, dim=0, keepdim=True)
        tree_word = torch.tensor(tree.word, device=device).view(1, -1)
        tree_index = torch.tensor(tree.index, device=device)
        # print(self.Embedding(tree_index).shape, tree_word.shape)
        inputVector = tree_word.mm(self.Embedding(tree_index))
        iou = self.ioux(inputVector) + self.iouh(child_h_Sum)
        r, z = torch.split(iou, iou.size(1) // 2, dim=1)  # 取出三个门 i 输入门， o输出门
        r, z = torch.sigmoid(r), torch.sigmoid(z)
        hc = torch.tanh(torch.matmul(inputVector, self.wx) + torch.matmul(r * child_h_Sum, self.wh) + self.b)
        h = z * child_h_Sum + (1-z) *  hc
        return h

    def forward(self, tree):  # 求出结果。
        for idx in range(len(tree.children)):
            self.forward(tree.children[idx])  # 对于树要递归。

        if len(tree.children) == 0:
            # child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            # child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = torch.tensor(tree.word, device=device).detach().new(1, self.mem_dim).fill_(
                0.).requires_grad_()
        else:
            # print(map(lambda x: x.state, tree.children))
            # child_h = torch.tensor(list(map(lambda x: x.state, tree.children)))
            child_h = list(map(lambda x: x.state, tree.children))
            child_h = torch.cat(child_h, dim=0)

        # tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        tree.state = self.node_forward(tree, child_h)
        output = self.fc(tree.state)
        return output


    ####################################################################
# # 建立rumorDataset
# class RumorDataset(torch.utils.data.Dataset):
#     def __init__(self, treex, treey):
#         super(RumorDataset, self).__init__()
#         self.treex = treex
#         self.treey = treey
# 
#     def __getitem__(self, index):
#         return (self.treex[index], self.treey[index])
#     def __len__(self):
#         return len(self.treex)
#     def loss_fn(self, y, pred_y):
#         return T.sum(T.sqr(y - pred_y))

    '''def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)'''

