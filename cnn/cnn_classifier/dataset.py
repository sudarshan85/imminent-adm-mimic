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
  def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer) -> None:
    self._df = df
    self._vectorizer = vectorizer

    # get maximum sequence length and +2 to account for EOS and BOS
    self._max_seq_len = max(map(lambda context: len(context.split(' ')), self._df['scispacy_note'])) + 2

  @classmethod
  def load_data_and_create_vectorizer(cls, df: pd.DataFrame, min_freq: int):
    return cls(df, Vectorizer.from_dataframe(df, 'scispacy_note', min_freq))

  @classmethod
  def load_data_and_vectorizer_from_file(cls, df: pd.DataFrame, vectorizer_path: Union[Path, str]):
    vectorizer_path = Path(vectorizer_path)
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(df, vectorizer)

  @classmethod
  def load_data_and_vectorizer(cls, df: pd.DataFrame, vectorizer: Vectorizer):
    return cls(df, vectorizer)

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
    note_vector = np.asarray(self._vectorizer.vectorize(row['scispacy_note'], self._max_seq_len))
    class_label = np.asarray(row['class_label'], dtype=np.float32)

    return (note_vector, class_label)

  def get_label(self, idx: int) -> int:
    return self._df.iloc[idx]['class_label']

  @property
  def vectorizer(self):
    return self._vectorizer

  @property
  def max_seq_length(self):
    return self._max_seq_len

  def __len__(self):
    return len(self._df)