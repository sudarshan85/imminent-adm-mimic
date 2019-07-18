#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')

args = Namespace(
  workdir=path/'workdir',
  figdir=path/'figures',
  raw_csv=path/'raw_dataset.csv',
  proc_csv=path/'proc_dataset.csv',
  imminent_adm_cols=['hadm_id', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'prolonged_stay_label'],
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime'],
  dates=['charttime', 'intime'],
  ia_thresh={
    'lr': 0.45,
    'rf': 0.27,
    'gbm': 0.435,
    'nn': 0.2,
    },
  ps_thresh={
    'lr': 0.39,
    'rf': 0.36,
    'gbm': 0.324,
    'nn': 0.27,
    },
)