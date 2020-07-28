#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict

from EEGNet import EEGNet
from run_model import run, draw_figure


# In[2]:


from dataloader import read_bci_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


# Load data
train_data, train_label, test_data, test_label = read_bci_data()

# Convert data type into float32
train_data  = torch.from_numpy(train_data).type(torch.float32).to(device)
train_label = torch.from_numpy(train_label).type(torch.long).to(device)
test_data   = torch.from_numpy(test_data).type(torch.float32).to(device)
test_label  = torch.from_numpy(test_label).type(torch.long).to(device)


# # Run Model

# In[4]:


net_list = {}
line_list = []
label_list = ['elu_train','elu_test','leaky_relu_train','leaky_relu_test','relu_train','relu_test']
#label_list = ['leaky_relu_train','leaky_relu_test']


# In[5]:


# ELU
net = EEGNet(hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
net_list['ELU'] = net
line_list.append(acc_train_list)
line_list.append(acc_test_list)


# In[6]:


# Leaky_relu
net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
net_list['Leaky_ReLU'] = net
line_list.append(acc_train_list)
line_list.append(acc_test_list)


# In[7]:


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

# In[8]:


plt.figure(figsize=[24,18])
draw_figure(plt,line_list,label_list, loc='best') 


# In[16]:


for i in {1,3,5}:
    print('Test Accuracy = ',line_list[i][-1])


# In[23]:


print('Test Accuracy:')
print('-------------------------')
print('With ELU        :',line_list[1][-1],'%')
print('With Leaky ReLU :',line_list[3][-1],'%')
print('With ReLU       :',line_list[5][-1],'%')


# # Experiment on weight decay

# In[24]:


# Leaky_relu
net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=0,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#optimizer = optim.Adagrad(net.parameters(),lr=1e-4,lr_decay=0.8,weight_decay=5e-3)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)


# In[25]:


print('Test Accuracy:')
print('-------------------------')
print('With weight-decay    :',line_list[3][-1],'%')
print('Without weight-decay :',acc_test_list[-1],'%')


# # Experiment on optimizer

# In[26]:


opt_acc_list = []


# In[27]:


net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-2, weight_decay=5e-2,momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
opt_acc_list.append(acc_test_list[-1])


# In[28]:


net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.Adam(net.parameters(),lr=1e-2, weight_decay=5e-2)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
opt_acc_list.append(acc_test_list[-1])


# In[29]:


net = EEGNet(act_f='leakyrelu',hyper=6).to(device)
optimizer = optim.Adagrad(net.parameters(),lr=1e-2, weight_decay=5e-2)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_list, acc_train_list, acc_test_list = run(net, train_data,train_label,test_data,test_label,                                               optimizer = optimizer, scheduler=lr_sch, num_epochs = 300, batch_size = 64,                                               print_freq = 500)
opt_acc_list.append(acc_test_list[-1])


# In[31]:


print('Test Accuracy:')
print('-------------------------')
print('Use SGD     :',opt_acc_list[0],'%')
print('Use Adam    :',opt_acc_list[1],'%')
print('Use Adagrad :',opt_acc_list[2],'%')


# In[ ]:




