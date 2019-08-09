#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/gbm'

args = Namespace(
  workdir=workdir,
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime', 'chartinterval'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  modeldir=workdir/'models',
  min_freq=3,
  ia_thresh=0.42,
  ps_thresh=0.34,
  start_seed=127,
  )

ia_params = {
  "objective": "binary",
  "metric": "binary_logloss",
  "bagging_fraction": 0.5,
  "bagging_freq": 5,
  "boosting": "dart",
  "feature_fraction": 0.5,
  "is_unbalance": True,
  "learning_rate": 0.1,
  "min_data_in_leaf": 3,
  "num_iterations": 150,
  "num_leaves": 50,
  "n_threads": 32,
}

ps_params = {
  "objective": "binary",
  "metric": "binary_logloss",
  "is_unbalance": True,
  "bagging_fraction": 0.7,
  "bagging_freq": 6,
  "boosting": "gbdt",
  "feature_fraction": 0.5,
  "learning_rate": 0.25,
  "min_data_in_leaf": 5,
  "num_iterations": 172,
  "n_threads": 32,
}