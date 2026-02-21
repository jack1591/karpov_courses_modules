#!/usr/bin/env python
# coding: utf-8

# In[2]:


def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    return (in_channels*kernel_size**2+(bias==True))*out_channels


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from IPython.display import clear_output
from matplotlib import cm
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from PIL import Image


# In[19]:


def train(model: nn.Module):
    model.train()

    total_loss = 0

    for x,y in tqdm(train_loader, desc = 'Train'):
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        total_loss = loss.item()

        loss.backward()

        optimizer.step()
    
    total_loss /= len(train_loader)
    return train_loader


# In[30]:


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc = 'Evaluation'):
        output = model(x)

        loss = loss_fn(output, y)
        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total+=y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct/total
    
    return total_loss, accuracy


# In[21]:


def create_mlp_model() -> nn.Module:
    my_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return my_model
    


# In[22]:


mnist_train = MNIST(
    '../datasets/mnist',
    train = True,
    download = True,
    transform = T.ToTensor()
)


# In[23]:


mnist_valid = MNIST(
    '../datasets/mnist',
    train = False,
    download = True,
    transform = T.ToTensor()
)


# In[24]:


train_loader = DataLoader(mnist_train, batch_size = 64, shuffle = True)
valid_loader = DataLoader(mnist_valid, batch_size = 64, shuffle = False)


# In[31]:


my_model = create_mlp_model()
num_epochs = 6
optimizer = torch.optim.Adam(my_model.parameters(), lr = 1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    train_loss = train(my_model)

    valid_loss, valid_accuracy = evaluate(my_model, valid_loader)

torch.save(my_model.state_dict(), 'model_weights.pt')


# In[ ]:




