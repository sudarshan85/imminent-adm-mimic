#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/gbm'

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
  imminent_threshold=0.3,
  discharge_threshold=0.39,
  start_seed=127,
  )
