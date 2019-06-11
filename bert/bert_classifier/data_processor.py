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
  text: str
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
    text = row[txt_col]
    label = row[label_col]
    examples.append(InputExample(eid=eid, text=text, label=label))

  return examples

def convert_examples_to_features(examples: List[InputExample], label_list: List[Union[int, str]], max_seq_len: int, tokenizer: BertTokenizer, is_pred=False) -> List[InputFeatures]:
  """
    Loads a data file into a list
  """
  features = []
  n_trunc = 0
  for example in tqdm(examples):
    tokens = tokenizer.tokenize(example.text)
    
    # Account for [CLS], [SEP] with -2
    if len(tokens) > max_seq_len - 2:
      logger.debug(f"Sample with ID '{example.eid}' has sequence length {len(tokens)} greater than max_seq_len ({max_seq_len}). Truncating sequence.")      
      tokens = tokens[:(max_seq_len - 2)]
      n_trunc += 1

    tokens = ['[CLS]'] + tokens + ['[SEP]']
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

  logger.warn(f"{(n_trunc/len(examples))*100:0.1f}% ({n_trunc}) of total examples have sequence length longer than max_seq_len ({max_seq_len})")
  return features

# @dataclass
# class DataContainer:
#   tokenizer: BertTokenizer,
#   df: pd.DataFrame,
#   txt_col: str,
#   label_col: str,
#   labels: List[Union[int, str]],
#   max_seq_len: int,
#   bs: int,
#   n_epochs: int,

#   def __post_init__(self):
#     self.train_ex = read_df(df.loc[(df['split'] == 'train')], txt_col, label_col)
#     self.train_feats = convert_examples_to_features(train_ex, labels, max_seq_len, tokenizer)

#     self._create_dataset()
#     self._create_dataloaders()

#     self.n_steps = (len(self.train_ds) // bs) * n_epochs

#   def _create_dataset(self):
#     input_ids = torch.tensor([f.input_ids for f in train_feats], dtype=torch.long)
#     input_mask = torch.tensor([f.input_mask for f in train_feats], dtype=torch.long)
#     segment_ids = torch.tensor([f.segment_ids for f in train_feats], dtype=torch.long)
#     label_ids = torch.tensor([f.label_id for f in train_feats], dtype=torch.long)
#     self.train_ds = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

#   def _create_dataloaders(self):
#     self.train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=bs)

# @dataclass
# class ModelContainer:
#   model: BertForSequenceClassification,
#   lr: float,
#   warmup_prop: float,
#   wd: float,
#   n_steps: int,
#   schedule: str='warmup_linear',

#   def __post_init__(self):
#     self.optimizer = build_optimizer(list(model.named_parameters()), n_steps, lr, warmup_prop, wd, schedule)