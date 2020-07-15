import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class BetaWarmup():
    def __init__(self, num_epochs=200,start=0, end=1, n_cycle=4, ratio=0.4, beta_type = 'cosine', maxi_val=1, latest_0 = 5):
        
        if beta_type =='linear': 
            self.schedule = self.frange_cycle_linear(start, end, num_epochs, n_cycle, ratio)*maxi_val
        elif beta_type =='sigmoid':
            self.schedule = self.frange_cycle_sigmoid(start, end, num_epochs, n_cycle, ratio)*maxi_val
        elif beta_type =='initial':
            self.schedule = self.initial(num_epochs)*maxi_val
        elif beta_type =='cosine':
            self.schedule = self.frange_cycle_cosine(start, end, num_epochs, n_cycle, ratio)*maxi_val

        
    def __getitem__(self, item):
        try:
        #return 0.01
            return self.schedule[item]
        except:
            return 1

    def initial(self, n_epoch):
        L = np.linspace(0, 1, n_epoch)
        return L

    def frange_cycle_linear(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L    

    def frange_cycle_sigmoid(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
                v += step
                i += 1
        return L    


    #  function  = 1 − cos(a), where a scans from 0 to pi/2

    def frange_cycle_cosine(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        # transform into [0, pi] for plots: 

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
                v += step
                i += 1
        return L    


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, num_colors=1, height=8, width=4096,
                hidden_size = 60):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
              nn.ReLU(),
              nn.Conv2d(num_filters, num_filters//2, kernel_size=(3,3), padding=1),
              nn.BatchNorm2d(num_filters//2),
              nn.ReLU(),
              nn.Conv2d(num_filters//2, num_filters, kernel_size=(3,3), padding=1)
           )

    def forward(self, x):
        return self.layers(x) + x


class ResidualStack(nn.Module):
    def __init__(self, num_filters, num_blocks=5):
        super(ResidualStack, self).__init__()

        self.layers = nn.Sequential(*[ResidualBlock(num_filters)
                                      for _ in range(num_blocks)])

    def forward(self, x):
        return F.relu(self.layers(x))
