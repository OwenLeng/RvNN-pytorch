# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: bottom-up recursive neural networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import os
from pytorch_treeRvNN import *

import torch
import torch.nn as nn
import time
import datetime
import random
from evaluate import *
import torch.utils.data as Data
from sklearn.metrics import classification_report

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# from Util import *

obj = "Twitter15"  # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "3"  # fold index, choose from 0-4
tag = ""
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 300
lr = 0.005

unit = "BU_RvNN-" + obj + str(fold) + '-vol.' + str(vocabulary_size) + tag
# lossPath = "../loss/loss-"+unit+".txt"
# modelPath = "../param/param-"+unit+".npz"

treePath = '../resource/data.BU_RvNN.vol_' + str(vocabulary_size) + tag + '.txt'

trainPath = "../nfold/RNNtrainSet_" + obj + str(fold) + "_tree.txt"
testPath = "../nfold/RNNtestSet_" + obj + str(fold) + "_tree.txt"
labelPath = "../resource/" + obj + "_label_All.txt"


# floss = open(lossPath, 'a+')

################################### tools #####################################
def str2matrix(Str, MaxL):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [0 for i in range(MaxL - l)]
    wordFreq += ladd
    wordIndex += ladd

    return wordFreq, wordIndex


def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    # if label in labelset_nonR:
    #     y_train = [1, 0, 0, 0]
    #     l1 += 1
    # if label in labelset_f:
    #     y_train = [0, 1, 0, 0]
    #     l2 += 1
    # if label in labelset_t:
    #     y_train = [0, 0, 1, 0]
    #     l3 += 1
    # if label in labelset_u:
    #     y_train = [0, 0, 0, 1]
    #     l4 += 1
    if label in labelset_nonR:
        y_train = 0
        l1 += 1
    if label in labelset_f:
        y_train = 1
        l2 += 1
    if label in labelset_t:
        y_train = 2
        l3 += 1
    if label in labelset_u:
        y_train = 3
        l4 += 1
    return y_train, l1, l2, l3, l4


def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'], tree[j]['maxL'])
        # print tree[j]['maxL']
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ## 
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
    ## 3. convert tree to DNN input
    # degree = tree[j]['max_degree']
    # x_word, x_index, tree = tree_gru.gen_nn_inputs(root, max_degree=degree, only_leaves_have_vals=False)
    return root


################################# load data ###################################
def loadData():
    print("loading tree label")
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
    print(len(labelDic))

    print("reading tree")  ## X
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        print(line.split('\t'))
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not eid in treeDic:
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    print("loading train set")
    tree_train, word_train, index_train, y_train, c = [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(trainPath):
        # if c > 8: break
        eid = eid.rstrip()
        if not eid in labelDic: continue
        if not eid in treeDic: continue
        if len(treeDic[eid]) < 2: continue
        ## 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        ## 2. construct tree)
        root = constructTree(treeDic[eid])
        tree_train.append(root)

        c += 1
    print(l1, l2, l3, l4)

    print("loading test set")
    tree_test, word_test, index_test, y_test, c = [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(testPath):
        # if c > 4: break
        eid = eid.rstrip()
        if not eid in labelDic: continue
        if not eid in treeDic: continue
        if len(treeDic[eid]) < 2: continue
        ## 1. load label        
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        ## 2. construct tree
        root = constructTree(treeDic[eid])
        tree_test.append(root)
        c += 1
    print(l1, l2, l3, l4)
    # print("train no:", len(tree_train), len(word_train), len(index_train), len(y_train))
    # print("test no:", len(tree_test), len(word_test), len(index_test), len(y_test))
    # print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
    # print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0])
    # exit(0)
    # return tree_train, word_train, index_train, y_train, tree_test, word_test, index_test, y_test
    # tree_train = torch.tensor(tree_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    # tree_test = torch.tensor(tree_test, device=device)
    y_test  = torch.tensor(y_test, device=device)
 
    return tree_train, y_train, tree_test, y_test


##################################### MAIN ####################################
## 1. load tree & word & index & label
# tree_train, word_train, index_train, y_train, tree_test, word_test, index_test, y_test = loadData()
tree_train, y_train, tree_test, y_test = loadData()

##### 2. 加载 dataloader
# trainDataset = RumorDataset(tree_train, y_train)
# testDataset = RumorDataset(tree_test, y_test)
# batch_size = 30
# num_workers = 0 if sys.platform.startswith('win32') else 4
# train_data_iter = Data.DataLoader(trainDataset, batch_size, shuffle=True, num_workers=num_workers)
# test_data_iter = Data.DataLoader(testDataset, batch_size, shuffle=True, num_workers=num_workers)
# print(trainDataset[0])
# print(torch.tensor(list(train_data_iter)).shape)

## 3. ini RNN model
t0 = time.time()
model = ChildSumTreeGRU(vocabulary_size, hidden_dim, hidden_dim, Nclass).to(device)
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
t1 = time.time()
print('Recursive model established,', (t1 - t0) / 60)

target_names = ['news-non-rumor', 'false', 'true', 'unverified'] #### 分类类别
## 4. train
losses_5, losses = [], []
num_examples_seen = 0

for epoch in range(Nepoch):
    ## one SGD
    indexs = [i for i in range(len(y_train))]
    random.shuffle(indexs)
    for i in indexs:
        pred_y = model(tree_train[i])
        # print(pred_y.shape, y_train[i].shape)
        loss = lossfn(pred_y, y_train[i].unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        # print loss, pred_y
        optimizer.step()
        losses.append(loss)
        num_examples_seen += 1
    print("epoch=%d: loss=%f" % (epoch, torch.mean(torch.tensor(losses))))
    sys.stdout.flush()
    ## cal loss & evaluate
    if epoch % 5 == 0:
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, torch.mean(torch.tensor(losses))))
        sys.stdout.flush()
        prediction = []
        for j in range(len(y_test)):
            prediction.append(model(tree_test[j]).argmax().cpu())
        # print(y_test.cpu(), prediction)
        res = classification_report(y_test.cpu(), prediction, target_names=target_names)
        print('results:')
        print(res)
    sys.stdout.flush()
    losses = []


