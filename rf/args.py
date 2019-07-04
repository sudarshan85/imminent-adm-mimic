#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/rf'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  modeldir=workdir/'models',
  min_freq=3,
  ia_thresh=0.32,
  ps_thresh=0.35,
  start_seed=127,
  )

ia_params = {
"class_weight": 'balanced',
"criterion": "entropy",
"max_features": 0.12,
"min_samples_leaf": 3,
"min_samples_split": 7,
"n_estimators": 108,
"oob_score": True,
}

ps_params = {}