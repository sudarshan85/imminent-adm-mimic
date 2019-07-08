#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')

args = Namespace(
  workdir=path/'workdir',
  figdir=path/'workdir/figdir',
  raw_csv=path/'raw_dataset.csv',
  proc_csv=path/'proc_dataset.csv',
  imminent_adm_cols=['hadm_id', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'prolonged_stay_label'],
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime'],
  dates=['charttime', 'intime'],
  ia_thresh={
    'lr': 0.5,
    'rf': 0.32,
    'gbm': 0.52,
    # 'cnn': 0.2,
    },
  ps_thresh={
    'lr': 0.47,
    'rf': 0.35,
    'gbm': 0.45,
    # 'cnn': 0.33,
    },
)