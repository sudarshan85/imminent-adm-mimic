#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/nn'

args = Namespace(
  workdir=workdir,
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime', 'chartinterval'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  device='cuda:2',
  start_seed=127,
  min_freq=3,
  batch_size=128,
  hidden_dim=100,
  dropout_p=0.1 ,
  lr=1e-3,
  wd=1e-3,
  max_lr=1e-1,
  max_epochs=100,
  ia_thresh=0.2,
  ps_thresh=0.5,
)
