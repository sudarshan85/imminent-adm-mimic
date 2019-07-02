#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')

args = Namespace(
  workdir=path/'workdir',
  figdir=path/'workdir/figdir',
  raw_csv=path/'raw_dataset.csv',
  proc_csv=path/'proc_dataset.csv',
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  # cols=['imminent_adm_label', 'prolonged_stay_label', 'scispacy_note', 'charttime', 'intime'],
  # dates=['charttime', 'intime'],
  # imminent_threshold={
  #   'lr': 0.47,
  #   'rf': 0.32,
  #   'gbm': 0.3,
  #   },
  # discharge_threshold={
  #   'lr': 0.48,
  #   'rf': 0.37,
  #   'gbm': 0.39,
  #   },
)