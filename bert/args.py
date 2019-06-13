#!/usr/bin/env python

import torch
from argparse import Namespace
from pathlib import Path

bert_model = 'bert-base-mimic-cased'
path = Path('../data')
workdir = path/'work_dir/bert'

args = Namespace (
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  bert_model=bert_model,
  modeldir=workdir/'models',
  bert_dir=Path(f'../pretrained/pytorch-bert/{bert_model}'),
  max_seq_len=256,
  do_lower_case=False,
  bs=32,
  device='cuda:0',
  start_seed=127,
  cols=['class_label', 'note'],
  lr=5e-5,
  n_epochs=2,
  wd=0.1,
  warmup_prop=0.1,
  gradient_accumulation_steps=1,
  loss_scale=0,
  do_train=True,
  do_eval=True,
  bc_threshold=0.1,
  num_labels=1,
  labels=[0, 1],
  n_gpu=torch.cuda.device_count(),
)
