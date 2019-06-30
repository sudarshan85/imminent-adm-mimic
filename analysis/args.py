#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
figdir=path/'workdir/figures'

args = Namespace(
  path=path,
  figdir=figdir,
  raw_csv=path/'mimic_icu_pred_raw_dataset.csv',
  proc_csv=path/'mimic_icu_pred_proc_dataset.csv',
  imminent_threshold={
    'lr': 0.47,
    'rf': 0.32,
    'gbm': 0.3,
    },
  discharge_threshold={
    'lr': 0.48,
    'rf': 0.37,
    'gbm': 0.39,
    },
)