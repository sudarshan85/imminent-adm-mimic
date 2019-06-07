#!/usr/bin/env python

import logging
import pandas as pd

from typing import Callable, Any
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

@dataclass
class ModelContainer:
  model: nn.Module
  loss_fn: Callable
  optimizer: optim
  reduce_lr: Callable = None

@dataclass
class DataContainer:
  df_with_splits: pd.DataFrame
  dataset_class: Any
  workdir: Path
  create_vec: bool = True
  bs: int = 32
  min_freq: int = 25
  with_test: bool = False
  weighted_sampling: bool = False

  def __post_init__(self):
    self._extract_from_splits()
    self._create_datasets()
    self._create_loaders()
    n_classes = self.df_with_splits['class_label'].nunique()
    self.n_classes = n_classes if n_classes > 2 else n_classes-1

  def _extract_from_splits(self):
    self.train_df = self.df_with_splits.loc[(self.df_with_splits['split'] == 'train')]
    if 'val' not in self.df_with_splits['split'].unique():
      logger.warn("No validation split provided. Using single sample of the training set.")
      self.val_df = self.train_df.sample()
    else:
      self.val_df = self.df_with_splits.loc[(self.df_with_splits['split'] == 'val')]

    self.df_lengths = {'train': len(self.train_df), 'val': len(self.val_df)}
    self.bss = {'train': self.bs, 'val': 2 * self.bs}
    self.n_batches = {'train': len(self.train_df) // self.bss['train'], 'val': len(self.val_df) // self.bss['val']}

    if self.with_test:
      self.test_df = self.df_with_splits.loc[(self.df_with_splits['split'] == 'test')]
      self.df_lengths['test'] = len(self.test_df)
      self.bss['test'] = self.df_lengths['test']
      self.n_batches['test'] = len(self.test_df) // self.bss['test']

  def _create_datasets(self):
    if self.create_vec:
      logger.info("Creating vectorizer...")
      self.train_ds = self.dataset_class.load_data_and_create_vectorizer(self.train_df, self.min_freq)
      self._vectorizer = self.train_ds.vectorizer
    else:
      try:
        self.train_ds = self.dataset_class.load_data_and_vectorizer_from_file(self.train_df, self.workdir)
        self._vectorizer = self.train_ds.vectorizer
      except FileNotFoundError:
        logger.info("Creating and saving vectorizer...")
        self.train_ds = self.dataset_class.load_data_and_create_vectorizer(self.train_df, self.min_freq)
        self.train_ds.save_vectorizer(self.workdir)
        self._vectorizer = self.train_ds.vectorizer
        logger.info("Finished!")

      self.train_ds = self.dataset_class.load_data_and_vectorizer_from_file(self.train_df, self.workdir)
      self._vectorizer = self.train_ds.vectorizer

    self.val_ds = self.dataset_class.load_data_and_vectorizer(self.val_df, self._vectorizer)

    if self.with_test:
      self.test_ds = self.dataset_class.load_data_and_vectorizer(self.test_df, self._vectorizer)

  def _create_loaders(self):
    if self.weighted_sampling:
      indices = list(range(len(self.train_ds)))
      n_samples = len(indices)
      counts = Counter([self.train_ds.get_label(idx) for idx in indices])
      weights = [1.0/counts[self.train_ds.get_label(idx)] for idx in indices]
      sampler = WeightedRandomSampler(weights, n_samples)
      self.train_dl = DataLoader(self.train_ds, self.bs, sampler=sampler, drop_last=True)
    else:
      self.train_dl = DataLoader(self.train_ds, self.bss['train'], shuffle=True, drop_last=True)

    self.val_dl = DataLoader(self.val_ds, self.bss['val'])

    if self.with_test:
      self.test_dl = DataLoader(self.test_ds, self.bss['test'])

  @property
  def vectorizer(self):
    return self._vectorizer

  def get_vocab_size(self):
    return len(self._vectorizer.vocab)

  def get_vocab_tokens(self):
    return self._vectorizer.vocab.idx_token_bidict.values()

  def get_batch_sizes(self):
    return self.bss

  def get_num_batches(self):
    return self.n_batches

  def get_dataset_size(self):
    return self.df_lengths

  def __repr__(self):
    s = f"DataContainer with {self.dataset_class.__name__}\n"
    s += f"Dataset Sizes: {self.df_lengths}\n"
    s += f"Batch size: {self.bs}\n"
    s += f"Number of batches: {self.n_batches}\n"
    s += f"Minimum Word Frequency: {self.min_freq}\n"
    s += f"Vocab size: {len(self._vectorizer.vocab)}\n"
    s += f"Number of classes: {self.n_classes}"
    return s
