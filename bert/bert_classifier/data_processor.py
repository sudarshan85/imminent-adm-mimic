#!/usr/bin/env python


import csv
import logging
import pandas as pd

from dataclasses import dataclass
from typing import List, Union
from pathlib import Path

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
  examples = []  
  for i, row in df.iterrows():
    eid = f'{set_type}-{i}'
    txt_a = row[txt_col]
    label = row[label_col]
    examples.append(InputExample(eid=eid, txt_a=txt_a, label=label))

  return examples
    