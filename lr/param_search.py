#!/usr/bin/env python

import logging
import datetime
import sys
import json
import warnings

import pandas as pd
import numpy as np

from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from utils.splits import set_group_splits
from args import args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

task = 'ia'

if __name__ == '__main__':
  seed = 42
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  if task == 'ia':
    task_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
    label = 'imminent_adm_label'
  if task == 'ps':
    tasl_df = ori_df[args.prolonged_stay_cols].copy()
    label = 'prolonged_stay_label'

  df = set_group_splits(task_df.copy(), group_col='hadm_id', seed=seed)
  vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, sublinear_tf=True, ngram_range=(2,2))

  x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note'])
  x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note'])
  y_train = df.loc[(df['split'] == 'train')][label].to_numpy()
  y_test = df.loc[(df['split'] == 'test')][label].to_numpy()

  params = {
    'class_weight': 'balanced',
    'solver': 'liblinear',
    'multi_class': 'ovr',
  }

  clf = LogisticRegression(**params)

  param_space = {
    'C': stats.uniform(0.1, 2),
    'dual': [True, False],
    'class_weight': ['balanced', None],
    'max_iter': stats.randint(100, 1000),
  }
    
  random_search = RandomizedSearchCV(clf, param_space, n_iter=100, cv=5, iid=False, verbose=1)

  logger.info("Starting random search...")
  t1 = datetime.datetime.now()
  random_search.fit(x_train, y_train)
  dt = datetime.datetime.now() - t1
  params_file = args.workdir/f'{task}_best_params.json'
  logger.info(f"Random search completed. Took {dt.seconds//3600} hours and {(dt.seconds//60)%60} minutes. Writing best params to {params_file}")
  json.dump(random_search.best_params_, params_file.open('w'))
