#!/usr/bin/env python

import logging
import datetime
import numpy as np
import sys
import warnings
import json
sys.path.append('../')
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import lightgbm

from utils.splits import set_group_splits
from args import args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

if __name__=='__main__':
  seed = 42
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  ia_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
  ps_df = ori_df[args.prolonged_stay_cols].copy()

  df = set_group_splits(ia_df.copy(), group_col='hadm_id', seed=seed)
  vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, sublinear_tf=True, ngram_range=(2,2))

  x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note'])
  x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note'])
  y_train = df.loc[(df['split'] == 'train')]['imminent_adm_label'].to_numpy()
  y_test = df.loc[(df['split'] == 'test')]['imminent_adm_label'].to_numpy()

  parameters = {
      'objective': 'binary',
      'metric': 'binary_logloss',
      'is_unbalance': 'true',
      'boosting': 'dart',
      'num_leaves': 50,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.75,
      'bagging_freq': 10,
  #     'learning_rate': 0.05,
      'num_threads': 32,
  #     'min_data_in_leaf': 3,
  #     'num_iterations': 100,
  }

  grid_params = {
    'num_leaves': [30, 40, 50, 60],
    'feature_faction': [0.25, 0.5, 0.75],
    'bagging_fraction': [0.25, 0.5, 0.75],
    'bagging_freq': [5, 10, 20],
    'learning_rate': [1e-3, 1e-2, 1e-1],
    'min_data_in_leaf': [3, 10, 18],
  }

  clf = lightgbm.LGBMClassifier(**parameters, verbose=-1)
  grid = GridSearchCV(clf, grid_params, cv=3, n_jobs=32, verbose=1)

  logger.info("Starting grid search...")
  t1 = datetime.datetime.now()
  grid.fit(x_train, y_train)
  dt = datetime.datetime.now() - t1
  logger.info(f"Grid search completed. Took {dt.seconds//3600} hours and {(dt.seconds//60)%60} minutes. Writing best params to {args.workdir/'best_params.json'}")
  json.dump(grid.best_params_, (args.workdir/'best_params.json').open('w'))

