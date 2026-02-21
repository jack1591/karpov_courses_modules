#!/usr/bin/env python
# coding: utf-8

# In[56]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Optimizer


# In[14]:


def create_model():
    net = nn.Sequential(
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    return net


# In[48]:


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)
        total_loss += loss
        loss.backward()

        print(f'{loss:.5f}')

        optimizer.step()
    return float(total_loss / len(data_loader))


# In[59]:


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()
    loss = 0
    for x, y in data_loader:
        with torch.no_grad():
            output = model(x)
            loss += loss_fn(output, y)
    return float(loss / len(data_loader))


# In[49]:


class CustomTaskNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(700, 500)
        self.linear2 = nn.Linear(500, 200)
        self.linear3 = nn.Linear(200, 10)

        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.activation(self.linear1(x))
        output = self.activation(self.linear2(output))
        output = self.linear3(output)

        return output


# In[50]:


model = create_model()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)


# In[51]:


from torch.utils.data import Dataset, TensorDataset


# In[52]:


n_features = 100
n_objects = 10

torch.manual_seed(0)

w_true = torch.randn(n_features)

X = (torch.rand(n_objects, n_features) - 0.5) * 10
X *= (torch.arange(n_features) * 3 + 2)
Y = (X @ w_true + torch.randn(n_objects)).unsqueeze(1)
dataset = TensorDataset(X, Y)


# In[53]:


from torch.utils.data import DataLoader
# делим выборку на батчи по 4 объекта
loader = DataLoader(dataset, batch_size = 4, shuffle = True)


# In[54]:


train(model, loader, optimizer, nn.MSELoss())


# In[60]:


evaluate(model, loader, nn.MSELoss())


# In[ ]:




