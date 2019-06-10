#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

bert_model = 'bert-base-mimic-cased'
path = Path('../data')
workdir = path/'workdir/bert'

args = Namespace (  
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  bert_model=bert_model,
  bert_dir=Path(f'../pretrained/pytorch-bert/{bert_model}'),
  max_seq_len=512,
  do_lower_case=False,
  bs=128,
  device='cuda:3',
  start_seed=127,
  cols=['class_label', 'note'],
  labels=[0, 1],
)
