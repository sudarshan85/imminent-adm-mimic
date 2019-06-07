#!/usr/bin/env python

import pandas as pd
import numpy as np
import string

from collections import Counter

from .vocabulary import Vocabulary, SequenceVocabulary

class Vectorizer(object):
  def __init__(self, vocab: Vocabulary):
    self.vocab = vocab

  def vectorize(self, note: str, vector_len: int) -> np.ndarray:
    """
      Args:
        note: string of words separated by a space representing a clinical note
        vector_len: an argument for forcing the length of index vector

      Returns:
        vectorized note
    """
    idxs = [self.vocab.lookup_token(token) for token in note.split(' ')]
    vector = [self.vocab.bos_idx] + idxs + [self.vocab.eos_idx]

    out_vector = np.zeros(vector_len, dtype=np.int64)
    out_vector[:len(vector)] = vector
    out_vector[len(vector):] = self.vocab.mask_idx

    return out_vector

  @classmethod
  def from_dataframe(cls, df: pd.DataFrame, col_name: str, min_freq: int):
    """
      Instantiate the vectorizer from dataset dataframe

      Args:
        df: target dataset

      Returns:
        an instance of the vectorizer
    """
    vocab = SequenceVocabulary()
    word_counts: Counter = Counter()
    for note in df[col_name]:
      for word in note.split(' '):
        word_counts[word] += 1

    for word, count in word_counts.items():
      if count > min_freq:
        vocab.add_token(word)

    return cls(vocab)

  @classmethod
  def from_serializable(cls, contents: dict):
    vocab = SequenceVocabulary.from_serializable(contents['vocab'])
    return cls(vocab)

  def to_serializable(self) -> dict:
    return {
        'vocab': self.vocab.to_serializable(),
        }
