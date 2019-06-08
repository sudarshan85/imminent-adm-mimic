#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'work_dir'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  temporal_pkl=path/'temporal_notes.pkl',
  min_freq=3,
  bc_threshold={
    'lr': 0.47,
    'rf': 0.32,
    },
  )
