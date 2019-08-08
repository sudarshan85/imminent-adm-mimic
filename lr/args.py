#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/lr'

args = Namespace(
  workdir=workdir,
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime', 'chartinterval'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  modeldir=workdir/'models',
  min_freq=3,
  ia_thresh=0.45,
  ps_thresh=0.41,
  start_seed=127,
  )

ia_params = {
  'class_weight': 'balanced',
  'solver': 'liblinear',
  'multi_class': 'ovr',
  'dual': True,
  'C': 0.336,
}  

ps_params = {
  'class_weight': 'balanced',
  'solver': 'liblinear',
  'multi_class': 'ovr',
  'dual': True,
}

