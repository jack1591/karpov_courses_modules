#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[13]:


def function01(tensor: torch.Tensor, count_over: str) -> torch.Tensor:
    if count_over == 'columns':
        return torch.mean(tensor, axis = 0)
    return torch.mean(tensor, axis = 1)


# In[25]:


def function02(tensor: torch.Tensor) -> torch.Tensor:
    n_features = tensor.shape[1]
    weights = torch.rand(n_features, dtype = torch.float32, requires_grad = True)
    return weights


# In[78]:


def function03(x: torch.Tensor, y: torch.Tensor):
    w = function02(x)
    step_size = 1e-2
    
    y_pred = torch.matmul(x, w)
    
    mse = torch.mean((y_pred - y) ** 2)

    while mse>=1:        
        mse.backward()
    
        with torch.no_grad():
            w -= w.grad * step_size
    
        w.grad.zero_()
    
        y_pred = torch.matmul(x, w)
    
        mse = torch.mean((y_pred - y) ** 2)

    return w


# In[79]:


def function04(x: torch.Tensor, y: torch.Tensor):
    step_size = 1e-2
    layer = nn.Linear(in_features = x.shape[1], out_features = 1)

    y_pred = layer(x).ravel()
        
    mse = torch.mean((y_pred - y.ravel()) ** 2)

    while mse>=0.3:        
        mse.backward()
    
        with torch.no_grad():
            layer.weight -= layer.weight.grad * step_size
            layer.bias -= layer.bias.grad * step_size
        
        layer.zero_grad()
    
        y_pred = layer(x).ravel()
        
        mse = torch.mean((y_pred - y) ** 2)
        print(mse)
    return layer

