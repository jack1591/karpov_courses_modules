#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
from torch.utils.data import DataLoader
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2):
    model.eval()
    logits_for_every_object_by_iteration = []
    with torch.no_grad():
        for iteration in range(iterations):
            logits_for_every_object = []
            for features, _ in loader:
                features = features.to(device)
                output = model(features)
                logits_for_every_object.append(output)    
            logits_for_every_object = torch.cat(logits_for_every_object, dim = 0)
            logits_for_every_object_by_iteration.append(logits_for_every_object.unsqueeze(-1))
    all_logits = torch.cat(logits_for_every_object_by_iteration, dim = -1)
    predictions = all_logits.mean(dim = -1)
    predictions = torch.argmax(predictions, dim = 1)
    return predictions

