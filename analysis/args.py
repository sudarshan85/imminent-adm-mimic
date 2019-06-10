#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'work_dir'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'full_processed_dataset.csv',
  min_freq=3,
  bc_threshold={
    'lr': 0.47,
    'rf': 0.32,
    'cnn': 0.23,
    },
  )
