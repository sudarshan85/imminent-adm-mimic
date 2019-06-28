#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'workdir/lr'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'mimic_icu_pred_proc_dataset.csv',
  cols=['imminent_label', 'discharge_label', 'scispacy_note'],
  full_run_cols=['imminent_label', 'discharge_label', 'scispacy_note', 'charttime', 'intime'],
  dates=['charttime', 'intime'],
  modeldir=workdir/'models',
  min_freq=3,
  imminent_threshold=0.47,
  discharge_threshold=0.48,
  start_seed=127,
  )

