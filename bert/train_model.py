#!/usr/bin/env python

import logging
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../')

from typing import List, Union, Tuple, Optional
from tqdm import tqdm, trange
from dataclasses import dataclass

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler

from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from args import args
from utils.splits import set_two_splits

@dataclass
class InputExample(object):
  """
    A single training/test example for simple sequence classification.
  """
  eid: str
  text: str
  label: int

@dataclass
class InputFeatures(object):
  """
    A single set of features representing the data.
  """
  input_ids: List[int]
  input_mask: List[int]
  segment_ids: List[int]
  label_id: List[Optional[int]]

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

def build_optimizer(named_params: List[Tuple[int, torch.nn.parameter.Parameter]], n_steps: int, lr: float, warmup_prop: float, wd: float, schedule: str='warmup_linear') -> BertAdam: 
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  grouped_params = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  
  return BertAdam(grouped_params, lr=lr, warmup=warmup_prop, t_total=n_steps, schedule=schedule, weight_decay=wd)


def get_sample(df, sample_pct=0.01, with_val=True, seed=None):
  train = df.loc[(df['split']) == 'train'].sample(frac=sample_pct, random_state=seed)
  train.reset_index(inplace=True, drop=True)

  if with_val:
    val = df.loc[(df['split']) == 'val'].sample(frac=sample_pct, random_state=seed)
    val.reset_index(inplace=True, drop=True)
    return pd.concat([train, val], axis=0) 

  return train  

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

  logger.warning(f"{(n_trunc/len(examples))*100:0.1f}% ({n_trunc}) of total examples have sequence length longer than max_seq_len ({max_seq_len})")
  return features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)


if __name__=='__main__':
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)
  df = get_sample(set_two_splits(ori_df.copy(), 'val'))
  tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.do_lower_case)
  train_ex = read_df(df.loc[(df['split'] == 'train')], 'note', 'class_label')
  labels = 1-df['class_label'].unique()
  train_feats = convert_examples_to_features(train_ex, labels, args.max_seq_len, tokenizer)
  model = BertForSequenceClassification.from_pretrained(args.bert_dir, num_labels=1)

  input_ids = torch.tensor([f.input_ids for f in train_feats], dtype=torch.long)
  input_mask = torch.tensor([f.input_mask for f in train_feats], dtype=torch.long)
  segment_ids = torch.tensor([f.segment_ids for f in train_feats], dtype=torch.long)
  label_ids = torch.tensor([f.label_id for f in train_feats], dtype=torch.long)

  train_ds = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
  train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=args.bs)

  n_steps = (len(train_ds)//args.bs) * args.n_epochs
  optimizer = build_optimizer(list(model.named_parameters()), n_steps, args.lr, args.warmup_prop, args.wd, args.schedule)

  logger.info("Done")
