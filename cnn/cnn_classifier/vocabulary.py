#!/usr/bin/env python

from typing import List
from bidict import bidict

class Vocabulary(object):
  def __init__(self, idx_token_bidict: bidict=None):
    if not idx_token_bidict:
      idx_token_bidict = bidict()

    self.idx_token_bidict = idx_token_bidict
    self.size = len(self.idx_token_bidict)

  def to_serializable(self):
    return {'idx_token_map': dict(self.idx_token_bidict)}

  @classmethod
  def from_serializable(cls, contents):
    idx_token_map = {int(k): v for k,v in contents['idx_token_map'].items()}
    idx_token_bidict = bidict(idx_token_map)
    return cls(idx_token_bidict)

  def add_token(self, token: str) -> int:
    if token in self.idx_token_bidict.values():
      idx = self.idx_token_bidict.inverse[token]
    else:
      idx = len(self.idx_token_bidict)
      self.idx_token_bidict.put(idx, token)
      self.size += 1

    return idx

  def add_many(self, tokens: List[str]) -> List[int]:
    return [self.add_token(token) for token in tokens]

  def lookup_token(self, token: str) -> int:
    return self.idx_token_bidict.inverse[token]

  def lookup_idx(self, idx: int) -> str:
    if idx not in self.idx_token_bidict:
      raise KeyError(f"The index {idx} is no in the vocabulary")
    return self.idx_token_bidict[idx]

  def __repr__(self):
    return f'<Vocabulary(size={self.size})'

  def __len__(self):
    return self.size

class SequenceVocabulary(Vocabulary):
  def __init__(self, idx_token_bidict: bidict=None, unk_token: str='<UNK>', mask_token:
      str='<MASK>', bos_token: str='<BOS>', eos_token: str='<EOS>'):
    super(SequenceVocabulary, self).__init__(idx_token_bidict)

    # save all the tokens
    self.mask = mask_token
    self.eos = eos_token
    self.bos = bos_token
    self.unk = unk_token

    # add all to the bidict
    self.mask_idx = self.add_token(self.mask)
    self.unk_idx = self.add_token(self.unk)
    self.bos_idx = self.add_token(self.bos)
    self.eos_idx = self.add_token(self.eos)

  def to_serializable(self):
    contents = super(SequenceVocabulary, self).to_serializable()
    contents.update({
        'unk_token': self.unk,
        'mask_token': self.mask,
        'bos_token': self.bos,
        'eos_token': self.eos,
      })
    return contents

  @classmethod
  def from_serializable(cls, contents):
    idx_token_map = {int(k): v for k,v in contents['idx_token_map'].items()}
    idx_token_bidict = bidict(idx_token_map)
    unk = contents['unk_token']
    mask = contents['mask_token']
    bos = contents['bos_token']
    eos = contents['eos_token']
    return cls(idx_token_bidict, unk, mask, bos, eos)

  def lookup_token(self, token: str) -> int:
    """
      Retrieve the index associated with the token or the UNK index if
      token isn't present

      Args:
        token: the token to lookup

      Returns:
        index: the index to the corresponding token

      Notes:
        The `unk_idx` must be >= 0 (having been added into hte vocabulary)
        for the UNK functionality
    """
    if self.unk_idx >= 0:
      if token in self.idx_token_bidict.values():
        return self.idx_token_bidict.inverse[token]
      else:
        return self.unk_idx
    else:
      return self.idx_token_bidict.inverse[token]