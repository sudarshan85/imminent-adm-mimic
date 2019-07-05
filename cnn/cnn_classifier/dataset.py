#!/usr/bin/env python

import pandas as pd
import numpy as np
import json
import torch

from typing import Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .vectorizer import Vectorizer

class NoteDataset(Dataset):
  def __init__(self, df: pd.DataFrame, label_col: str, vectorizer: Vectorizer) -> None:
    self._df = df
    self._vectorizer = vectorizer
    self.label_col = label_col

    # get maximum sequence length and +2 to account for EOS and BOS
    self._max_seq_len = max(map(lambda context: len(context.split(' ')), self._df['processed_note'])) + 2

  @classmethod
  def load_data_and_create_vectorizer(cls, df: pd.DataFrame, label_col: str, min_freq: int):
    return cls(df, label_col, Vectorizer.from_dataframe(df, 'processed_note', min_freq))

  @classmethod
  def load_data_and_vectorizer_from_file(cls, df: pd.DataFrame, label_col: str, vectorizer_path: Union[Path, str]):
    vectorizer_path = Path(vectorizer_path)
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(df, label_col, vectorizer)

  @classmethod
  def load_data_and_vectorizer(cls, df: pd.DataFrame, label_col: str, vectorizer: Vectorizer):
    return cls(df, label_col, vectorizer)

  @staticmethod
  def load_vectorizer(workdir: Union[Path, str]) -> Vectorizer:
    vectorizer_path = Path(workdir)/'vectorizer.json'
    with open(vectorizer_path) as fp:
      return Vectorizer.from_serializable(json.load(fp))

  def save_vectorizer(self, workdir: Union[Path, str]) -> None:
    vectorizer_path = Path(workdir)/'vectorizer.json'
    with open(vectorizer_path, 'w') as fp:
      json.dump(self._vectorizer.to_serializable(), fp)

  def __getitem__(self, idx):
    row = self._df.iloc[idx]
    note_vector = np.asarray(self._vectorizer.vectorize(row['processed_note'], self._max_seq_len))
    class_label = np.asarray(row[self.label_col], dtype=np.float32)

    return (note_vector, class_label)

  def get_label(self, idx: int) -> int:
    return self._df.iloc[idx][self.label_col]

  @property
  def vectorizer(self):
    return self._vectorizer

  @property
  def max_seq_length(self):
    return self._max_seq_len

  def __len__(self):
    return len(self._df)