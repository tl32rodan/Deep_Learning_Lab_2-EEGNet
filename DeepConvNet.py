import torch
import torch.nn as nn
from collections import OrderedDict

class DeepConvNet(nn.Sequential):
    def __init__(self,act_f = 'elu',*args, **kwargs):
        act_f_list = {'elu':nn.ELU,
                      'leakyrelu':nn.LeakyReLU,
                      'relu': nn.ReLU,
                      'prelu':nn.PReLU
                     }
        self.C = 2
        self.T = 750
        
        ### Layer 1
        firstConv = nn.Sequential(
            nn.Conv2d(1,25,kernel_size=(1,5)),
            nn.Conv2d(25,25,kernel_size=(self.C,1)),
            nn.BatchNorm2d(2*25),
            act_f_list[act_f](*args, **kwargs),# Activation function
            nn.MaxPool2d((1,2)),
            nn.Dropout2d()
        )
        
        ### Layer 2
        secondConv = nn.Sequential(
            nn.Conv2d(25,50,kernel_size=(1,5)),
            nn.BatchNorm2d(2*50),
            act_f_list[act_f](*args, **kwargs),# Activation function
            nn.MaxPool2d((1,2)),
            nn.Dropout2d()
        )
        
        ### Layer 3
        thirdConv = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,5)),
            nn.BatchNorm2d(2*100),
            act_f_list[act_f](*args, **kwargs),# Activation function
            nn.MaxPool2d((1,2)),
            nn.Dropout2d()
        )
        
        ### Layer 4
        fourthConv = nn.Sequential(
            nn.Conv2d(100,200,kernel_size=(1,5)),
            nn.BatchNorm2d(2*200),
            act_f_list[act_f](*args, **kwargs),# Activation function
            nn.MaxPool2d((1,2)),
            nn.Dropout2d(),
            nn.Flatten()
        )
        
        ### Fully connected layer
        classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        super(DeepConvNet, self).__init__(OrderedDict([
                  ('firstConv' , firstConv),
                  ('secondConv', secondConv),
                  ('thirdConv' , thirdConv),
                  ('fourthConv', fourthConv),
                  ('classify'  , classify)
                ]))
        
    def forward(self,x):
        x = super(DeepConvNet,self).forward(x)
        return x
    
    def infer_and_cal_acc(self,x,ground_truth):
        # Run self.forward(x) and calculate accuracy with ground_truth
        
        y = self.forward(x)
        # Get the calssification result of y,
        # because y has 2 channels for each class
        _, y_hat = torch.max(y,1)
        # Calculate accuracy
        acc = 100*(len(ground_truth[ground_truth==y_hat])/len(ground_truth))
        return acc