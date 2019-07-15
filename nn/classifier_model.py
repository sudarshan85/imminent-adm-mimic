#!/usr/bin/env python

import torch
from torch import nn
from torch.nn import functional as F

class CNNClassifier(nn.Module):
  def __init__(self, vocab_sz, emb_sz, hidden_dim, dropout_p, n_channels, embeddings=None):
    super(CNNClassifier, self).__init__()
    
    pretrained_emb = torch.from_numpy(embeddings).float()
    self.emb = nn.Embedding(vocab_sz, emb_sz, _weight=pretrained_emb, padding_idx=0)
    
    self.convnet = nn.Sequential(
      nn.Conv1d(in_channels=emb_sz, out_channels=n_channels, kernel_size=3),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3),
      nn.ELU()      
    )
    
    self.fc1 = nn.Linear(in_features=n_channels, out_features=hidden_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_p)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, x_in):
    x_emb = self.emb(x_in).permute(0, 2, 1)
    x_out = self.convnet(x_emb)
    remaining_size = x_out.size(dim=2)
    
    x_out = F.avg_pool1d(x_out, remaining_size).squeeze(dim=2)
    x_out = self.dropout(x_out)
    x_out = self.relu(self.dropout(self.fc1(x_out)))
    x_out = self.fc2(x_out)
    
    return x_out.squeeze(1)


class NNClassifier(nn.Module):
  def __init__(self, vocab_sz, hidden_dim, dropout_p):
    super(NNClassifier, self).__init__()
    
    self.fc1 = nn.Linear(in_features=vocab_sz, out_features=hidden_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_p)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, x_in):
    x_out = self.fc1(x_in)
    x_out = self.dropout(self.relu(x_out))
    x_out = self.fc2(x_out)
    
    return x_out.squeeze(1)