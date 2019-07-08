#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/gbm'

args = Namespace(
  workdir=workdir,
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  modeldir=workdir/'models',
  min_freq=3,
  ia_thresh=0.526,
  ps_thresh=0.45,
  start_seed=127,
  )

ia_params = {
  "objective": "binary",
  "metric": "binary_logloss",
  "bagging_fraction": 0.413,
  "bagging_freq": 20,
  "boosting": "dart",
  "feature_fraction": 0.288,
  "is_unbalance": True,
  "lambda_l1": 3.367,
  "learning_rate": 0.037,
  "max_bin": 9,
  "min_data_in_leaf": 12,
  "num_iterations": 163,
  "num_leaves": 45,
}

ps_params = {
  "objective": "binary",
  "metric": "binary_logloss",  
  "boosting": "dart",
  "bagging_fraction": 0.7,
  "bagging_freq": 7,
  "feature_fraction": 0.4,
  "is_unbalance": True,
  "lambda_l1": 0.35,
  "learning_rate": 0.25,
  "max_bin": 150,
  "min_data_in_leaf": 3,
  "num_iterations": 105,
  "num_leaves": 50,
}