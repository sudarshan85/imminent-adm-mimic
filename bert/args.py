#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

bert_model = 'bert-base-mimic-cased'
path = Path('../data')
workdir = path/'work_dir/bert'

args = Namespace (  
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  bert_model=bert_model,
  bert_dir=Path(f'../pretrained/pytorch-bert/{bert_model}'),
  max_seq_len=256,
  do_lower_case=False,
  bs=128,
  device='cuda:0',
  start_seed=127,
  cols=['class_label', 'note'],
  lr=5e-5,
  n_epochs=1,
  wd=0.1,
  warmup_prop=0.1,
  schedule='warmup_linear',
)