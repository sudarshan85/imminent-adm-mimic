#!/usr/bin/env python

import torch
from torch import nn
from torch.nn import functional as F

class NoteClassifier(nn.Module):
  def __init__(self, emb_sz, vocab_size, n_channels, hidden_dim, n_classes, dropout_p, emb_dropout=None, pretrained=None, padding_idx=0):
    super(NoteClassifier, self).__init__()

    if pretrained is None:
      self.emb = nn.Embedding(vocab_size, emb_sz, padding_idx)
    else:
      pretrained_emb = torch.from_numpy(pretrained).float()
      self.emb = nn.Embedding(vocab_size, emb_sz, padding_idx, _weight=pretrained_emb)

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

    if emb_dropout is not None:
      self.emb_dropout = nn.Dropout(p=emb_dropout)
    else:
      self.emb_dropout = False
      
    self.dropout = nn.Dropout(p=dropout_p)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(in_features=n_channels, out_features=hidden_dim)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)

  def forward(self, x_in):
    # embed and permute so features are channels
    # conv1d (batch, channels, input)
    x_emb = self.emb(x_in).permute(0, 2, 1) if not self.emb_dropout else self.emb_dropout(self.emb(x_in).permute(0, 2, 1))
    # x_emb = self.emb_dropout(x_emb)
    features = self.convnet(x_emb)

    # average and remove extra dimension
    remaining_size = features.size(dim=2)
    features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
    features = self.dropout(features)

    # mlp classifier
    hidden_vector = self.fc1(features)
    hidden_vector = self.dropout(hidden_vector)
    hidden_vector = self.relu(hidden_vector)
    prediction_vector = self.fc2(hidden_vector)

    # return prediction_vector.squeeze(dim=1)
    return prediction_vector.squeeze(dim=1)
