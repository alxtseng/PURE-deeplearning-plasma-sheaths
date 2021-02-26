#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:02:16 2019

@author: alextseng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        self.input = nn.Linear(4, 512)
        self.hidden = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(512)
        self.output = nn.Linear(512, 500 * 180)
    
    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.output(x)
        return x


