#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


class Similarity1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        # encoder_states.shape = [T, N]
        # decoder_state.shape = [N]
        return encoder_states@decoder_state


# In[2]:


import torch
import torch.nn as nn


class Similarity2(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, intermediate_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(encoder_dim, intermediate_dim)
        self.fc2 = nn.Linear(decoder_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        # encoder_states.shape = [T, N]
        # decoder_state.shape = [N]
        # YOUR CODE HERE
        return self.fc3(self.tanh(self.fc1(encoder_states) + self.fc2(decoder_state)))


# In[ ]:




