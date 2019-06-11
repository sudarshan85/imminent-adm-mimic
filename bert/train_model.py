#!/usr/bin/env python

import logging
import sys
import numpy as np
import pandas as pd
import pickle
import torch
sys.path.append('../')

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler

from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from data_processor import convert_examples_to_features, read_df
from utils.splits import set_two_splits
from args import args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)


if __name__=='__main__':  
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)
  df = set_two_splits(ori_df.copy(), 'val')
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
