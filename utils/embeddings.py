#!/usr/bin/env

import logging
import sys
import numpy as np
import torch

from torch import nn
from typing import List
from pathlib import Path
from bidict import bidict
from annoy import AnnoyIndex

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)


class PretrainedEmbeddings(object):
  """
    A wrapper around pre-trained word vectors
  """
  def __init__(self, idx_token_bidict: bidict, pretrained_emb_matrix: np.ndarray, is_annoy:
      bool=False) -> None:
    """
      idx_word_map: a bidict mapping between indices and words
      pretrained_emb_matrix: list of numpy arrays
    """
    self.idx_token_bidict = idx_token_bidict
    self._pretrained_emb_matrix = pretrained_emb_matrix
    self._vocab_sz = len(pretrained_emb_matrix)
    self._emb_sz = pretrained_emb_matrix.shape[-1]
    self.is_annoy = is_annoy
    self._custom_emb_matrix = None

    if self.is_annoy:
      self.annoy_idx = AnnoyIndex(self._emb_sz, metric='euclidean')
      print("Building Annoy Index...")
      for i, _ in self.idx_token_bidict.items():
        self.annoy_idx.add_item(i, self._pretrained_emb_matrix[i])
      self.annoy_idx.build(50)
      print("Finished")

  def make_custom_embeddings(self, tokens: List[str]) -> None:
    """
      Create custom embedding matrix for a specific set of tokens

      Args:
        tokens: a list of tokens in the dataset
    """
    self._custom_emb_matrix = np.zeros((len(tokens), self._emb_sz))
    for idx, word in enumerate(tokens):
      if word in self.idx_token_bidict.values():
        self._custom_emb_matrix[idx, :] = self._pretrained_emb_matrix[
            self.idx_token_bidict.inverse[word]]
      else:
        embedding_i = torch.ones(1, self._emb_sz)
        torch.nn.init.xavier_uniform_(embedding_i)
        self._custom_emb_matrix[idx, :] = embedding_i

  @property
  def pretrained_embeddings(self):
    return self._pretrained_emb_matrix

  @property
  def custom_embeddings(self):
    return self._custom_emb_matrix

  @classmethod
  def from_file(cls, embedding_file: Path, is_annoy: bool=False, verbose=False):
    """
      Instantial from pre-trained vector file

      Vector file should be of the format:
        word0 x0_0 x0_1 ... x0_N
        word1 x1_0 x1_1 ... x1_N

      Args:
        embedding_file: location of the file
      Returns:
        Instance of PretrainedEmbeddings
    """
    logging_level = logging.INFO if verbose else logger.getEffectiveLevel()
    logger.setLevel(logging_level)
    idx_word_bidict = bidict()
    pretrained_emb_matrix = []

    logger.info(f"Loading pretrained embeddings from file {embedding_file}...")
    with open(embedding_file, 'r') as fp:
      for line in fp.readlines():
        line = line.split(' ')
        word = line[0]
        if line[-1] == '\n':
          del line[-1]
        vec = np.array([float(x) for x in line[1:]])

        idx_word_bidict.put(len(idx_word_bidict), word)
        pretrained_emb_matrix.append(vec)

    logger.info("Finished!")
    return cls(idx_word_bidict, np.stack(pretrained_emb_matrix), is_annoy=is_annoy)

  def get_embedding(self, word: str) -> np.ndarray:
    return self._pretrained_emb_matrix[self.idx_token_bidict.inverse[word]]

  @property
  def vocab_size(self) -> int:
    return self._vocab_sz

  def _get_neighbors(self, vector: np.ndarray, n: int=1) -> List[str]:
    """
      Given a vector, return its n nearest neighbors

      Args:
        vector: should match the size of the vectors in the Annoy idx
    """
    nn_idxs = self.annoy_idx.get_nns_by_vector(vector, n)
    return [self.idx_token_bidict[neighbor_idx] for neighbor_idx in nn_idxs]

  def __len__(self):
    return self._emb_sz

  def get_analogy(self, word1, word2, word3) -> str:
    """
      Computes solutions to analogies using word embeddings

      Analogies are word1 is to word2 as word3 is to ____
    """
    if not self.is_annoy:
      raise NameError(f"AnnoyIndex not built! is_annoy is {self.is_annoy}!")

    # get embedding of 3 words
    vec1 = self.get_embedding(word1)
    vec2 = self.get_embedding(word2)
    vec3 = self.get_embedding(word3)

    # compute 4th word embedding
    spatial_relationship = vec2 - vec1
    vec4 = vec3 + spatial_relationship

    closest_words = self._get_neighbors(vec4, n=4)
    existing_words = set([word1, word2, word3])
    closest_words = [word for word in closest_words if word not in existing_words]

    if len(closest_words) == 0:
      return 'Could not find nearest neighbors for computed vector!'

    words = []
    for word4 in closest_words:
      words.append(f'{word1} : {word2} :: {word3} : {word4}')

    return '\n'.join(words)

  def __repr__(self):
    return f"Pretrained Embeddings of {self._emb_sz} dimensions with {self._vocab_sz} words"

if __name__=='__main__':
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} path_to_embedding_file")
    sys.exit(-1)

  embedding_file = Path(sys.argv[1])
  embeddings = PretrainedEmbeddings.from_file(embedding_file, is_annoy=True)
  print(embeddings)
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'he', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'uncle', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['talk', 'communicate', 'read']))
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'king', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['king', 'queen', 'husband']))
