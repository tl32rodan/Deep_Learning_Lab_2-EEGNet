import torch
import torch.nn as nn
from collections import OrderedDict

class EEGNet(nn.Sequential):
    def __init__(self,act_f = 'elu',*args, **kwargs):
        act_f_list = {'elu':nn.ELU,
                      'leakyrelu':nn.LeakyReLU,
                      'relu': nn.ReLU,
                      'prerelu':nn.PReLU
                     }
        
        ### Layer 1
        firstconv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(1,51),stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16)
        )
        
        ### Layer 2
        depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32),
            # nn.ELU(alpha=0.1),
            # Change into programmable activation functions
            act_f_list[act_f](*args, **kwargs),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout2d(p = 0.25)
        )
        
        ### Layer 3
        seperableConv = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32),
            # nn.ELU(alpha=0.1),
            # Change into programmable activation functions
            act_f_list[act_f](*args, **kwargs),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout2d(p = 0.25),
            nn.Flatten()
        )
        
        ### Fully connected layer
        classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        super(EEGNet, self).__init__(OrderedDict([
                  ('firstconv'    , firstconv),
                  ('depthwiseConv', depthwiseConv),
                  ('seperableConv', seperableConv),
                  ('classify'     , classify)
                ]))
        
    def forward(self,x):
        x = super(EEGNet,self).forward(x)
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
        