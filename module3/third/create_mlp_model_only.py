#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
def create_mlp_model() -> nn.Module:
    my_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return my_model


# In[ ]:




