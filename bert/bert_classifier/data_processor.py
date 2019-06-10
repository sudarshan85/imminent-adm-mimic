#!/usr/bin/env python


import csv
import logging
import pandas as pd

from dataclasses import dataclass
from typing import List, Union
from pathlib import Path
from tqdm import tqdm_notebook as tqdm

from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

@dataclass
class InputExample(object):
  """
    A single training/test example for simple sequence classification.
  """
  eid: int
  txt_a: str
  txt_b: str=None
  label: str=None

@dataclass
class InputFeatures(object):
  """
    A single set of features representing the data.
  """
  input_ids: List[int]
  input_mask: List[int]
  segment_ids: List[int]
  label_id: List[int]

def read_df(df: pd.DataFrame, txt_col, label_col, set_type='train') -> List[InputExample]:
  """
    Function to read a dataframe with text and label and create a list
    of InputExample
  """
  logger.debug(f"Reading column text column {txt_col} and label column {label_col}")
  examples = []  
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    eid = f'{set_type}-{i}'
    txt_a = row[txt_col]
    label = row[label_col]
    examples.append(InputExample(eid=eid, txt_a=txt_a, label=label))

  return examples

def convert_examples_to_features(examples: List[InputExample], label_list: List[Union[int, str]], max_seq_len: int, tokenizer: BertTokenizer, is_pred=False) -> List[InputFeatures]:
  """
    Loads a data file into a list
  """
  features = []
  n_trunc = 0
  for example in tqdm(examples):
    tokens_a = tokenizer.tokenize(example.txt_a)
    
    # Account for [CLS], [SEP] with -2
    if len(tokens_a) > max_seq_len - 2:
      logger.debug(f"Sample with ID '{example.eid}' has sequence length {len(tokens_a)} greater than max_seq_len ({max_seq_len}). Truncating sequence.")      
      tokens_a = tokens_a[:(max_seq_len - 2)]
      n_trunc += 1

    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are 
    # attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    if is_pred:
      label_id = None
    else:
      label_id = example.label
    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))

  logger.warn(f"{(n_trunc/len(examples))*100:0.1f} ({n_trunc}) of total examples have sequence length longer than max_seq_len ({max_seq_len})")
  return features   