#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'work_dir/lr'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  cols=['class_label', 'scispacy_note'],
  modeldir=workdir/'models',
  min_freq=3,
  bc_threshold=0.47,
  start_seed=127,
  )
