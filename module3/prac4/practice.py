#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import torch
import torchvision.transforms as T


# In[49]:


def get_normalize(features: torch.Tensor):
    means = features.mean(axis = (0, 2, 3))
    stds = features.std(axis = (0, 2, 3))
    return (means, stds)


# In[56]:


'''
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
data_transform = T.Compose(
    [
        T.ToTensor()
    ]
)
train_data = CIFAR10('../datasets', train = True, transform = data_transform)
print(train_data.data.shape)
means, stds = get_normalize(train_data.data)
print(means)
print(stds)
'''


# In[52]:


def get_augmentations(train: bool = True) -> T.Compose:
    means = [125.3069180, 122.95039414, 113.86538318]
    stds = [62.99321928, 62.08870764, 66.70489964]
    if train:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomRotation(degrees = 20),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean = means, std = stds)
            ]
        )
    return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean = means, std = stds)
            ]
    )


# In[59]:


from torch.utils.data import DataLoader
from torch import nn
@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions = []
    model.eval()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, y_pred = torch.max(output, dim = 1)
        predictions.append(y_pred)
    return torch.cat(predictions, dim = 0)

