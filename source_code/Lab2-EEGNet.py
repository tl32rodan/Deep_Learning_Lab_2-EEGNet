#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict

from EEGNet import EEGNet
from run_model import run, draw_figure


# %%


from dataloader import read_bci_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%


# Load data
train_data, train_label, test_data, test_label = read_bci_data()

# Convert data type into float32
train_data  = torch.from_numpy(train_data).type(torch.float32).to(device)
train_label = torch.from_numpy(train_label).type(torch.long).to(device)
test_data   = torch.from_numpy(test_data).type(torch.float32).to(device)
test_label  = torch.from_numpy(test_label).type(torch.long).to(device)


# # Run Model

# %%


net_list = {}
line_list = []
label_list = ['elu_train','elu_test','leaky_relu_train','leaky_relu_test','relu_train','relu_test']
#label_list = ['leaky_relu_train','leaky_relu_test']


# %%


# ELU
net = EEGNet(hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
net_list['ELU'] = net
line_list.append(acc_train_list)
line_list.append(acc_test_list)


# %%


# Leaky_relu
net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
net_list['Leaky_ReLU'] = net
line_list.append(acc_train_list)
line_list.append(acc_test_list)


# %%


# Relu
net = EEGNet(act_f='relu',hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
net_list['ReLU'] = net
line_list.append(acc_train_list)
line_list.append(acc_test_list)


# # Draw loss & accuracy figures

# %%


plt.figure(figsize=[24,18])
draw_figure(plt,line_list,label_list, loc='best') 

# %%


print('Test Accuracy:')
print('-------------------------')
print('With ELU        :',line_list[1][-1],'%')
print('With Leaky ReLU :',line_list[3][-1],'%')
print('With ReLU       :',line_list[5][-1],'%')

