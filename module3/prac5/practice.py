#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch


# In[18]:


from torchvision.models import alexnet, vgg11, googlenet, resnet18


# In[19]:


from torch import nn
def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool=True):
    model = nn.Module()
    if model_name == 'alexnet':
        model = alexnet(pretrained = pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg11':
        model = vgg11(pretrained = pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'googlenet':
        model = googlenet(pretrained = pretrained, aux_logits = True)
        model.aux1.fc2 = nn.Linear(1024, num_classes)
        model.aux2.fc2 = nn.Linear(1024, num_classes)
        model.fc = nn.Linear(1024, num_classes)
    else:
        model = resnet18(pretrained = pretrained)
        model.fc = nn.Linear(512, num_classes)
    return model


# In[23]:


from torch import nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1), # 32 x 32 x 64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), # 32 x 32 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 x 16 x 128
            nn.Dropout2d(p = 0.2)
        )

        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1), # 16 x 16 x 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1), # 16 x 16 x 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1), # 16 x 16 x 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8 x 8 x 256
        )
        
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1), # 8 x 8 x 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4 x 4 x 512
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1), # 4 x 4 x 512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1), # 4 x 4 x 512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.block9 = nn.Sequential(
            nn.MaxPool2d(4), # 1 x 1 x 512
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x) + x

        x = self.block5(x)

        x = self.block6(x)

        x = self.block7(x)

        x = self.block8(x) + x
        
        x = self.block9(x)
        return x


# In[25]:


def create_advanced_skip_connection_conv_cifar():
    return MyModel()

