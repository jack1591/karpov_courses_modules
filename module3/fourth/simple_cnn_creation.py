#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


def create_simple_conv_cifar():
    return nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1), # 32 x 32 x 32
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.MaxPool2d(2),
        nn.Dropout(p = 0.2),
        
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), # 16 x 16 x 64
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten(),
        
        nn.Linear(16 * 16 * 64, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.Dropout(p = 0.2),
        nn.ReLU(),
        
        nn.Linear(256, 10)
    )

