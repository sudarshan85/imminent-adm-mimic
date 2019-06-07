#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir=path/'work_dir/lr'

args = Namespace(
  path=path,
  workdir=workdir,
  dataset_csv=path/'processed_dataset.csv',
  modeldir=workdir/'models',
  min_freq=3,
  vectorizer_path=workdir/'tfidf_vectorizer.pkl',
  bc_threshold=0.47,
  )
