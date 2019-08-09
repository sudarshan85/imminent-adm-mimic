#!/usr/bin/env python

import torch
from torch import nn

class MLPModel(nn.Module):
  def __init__(self, vocab_sz, hidden_dim, dropout_p):
    super(MLPModel, self).__init__()
    
    self.fc1 = nn.Linear(in_features=vocab_sz, out_features=hidden_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_p)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, x_in):
    x_out = self.fc1(x_in)
    x_out = self.dropout(self.relu(x_out))
    x_out = self.fc2(x_out)
    
    return x_out.squeeze(1)
