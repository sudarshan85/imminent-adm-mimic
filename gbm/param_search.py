#!/usr/bin/env python

import logging
import datetime
import sys
import json
import warnings

sys.path.append('../')
warnings.filterwarnings("ignore")

import pandas as pd

from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
import lightgbm

from utils.splits import set_group_splits
from args import args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    logger.error(f"Usage: {sys.argv[0]} task_name (ia|ps)")
    sys.exit(1)

  task = sys.argv[1]
  if task != 'ia' or task != 'ps':
    logger.error("Task values are either ia (imminent admission) or ps (prolonged stay)")
    sys.exit(1)

  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  if task == 'ia':
    logger.info(f"Running hyperparameter search for Imminent Admission Prediction task")
    task_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
    label = 'imminent_adm_label'
  if task == 'ps':
    logger.info(f"Running hyperparameter search for Prolonged Stay Prediction task ")
    task_df = ori_df[args.prolonged_stay_cols].copy()
    label = 'prolonged_stay_label'

  df = set_group_splits(task_df.copy(), group_col='hadm_id', seed=42)
  vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, sublinear_tf=True, ngram_range=(2,2))

  x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note'])
  x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note'])
  y_train = df.loc[(df['split'] == 'train')][label].to_numpy()
  y_test = df.loc[(df['split'] == 'test')][label].to_numpy()

  clf_params = {
      'objective': 'binary',
      'metric': 'binary_logloss',
  }

  clf = lightgbm.LGBMClassifier(**clf_params)

  param_space = {
    'num_leaves': stats.randint(27, 101),
    'bagging_fraction': stats.uniform(0.2, 0.7),
    'learning_rate': stats.reciprocal(1e-3, 5e-1),
    'min_data_in_leaf': stats.randint(2, 20),
    'is_unbalance': [True, False],
    'max_bin': stats.randint(3, 100),
    'boosting': ['gbdt', 'dart'],
    'bagging_freq': stats.randint(3, 31),
    'max_depth': stats.randint(0, 11),
    'feature_fraction': stats.uniform(0.2, 0.7),
    'lambda_l1': stats.uniform(0, 10),
    'num_iterations': stats.randint(100, 200),
  }

  random_search = RandomizedSearchCV(clf, param_space, n_iter=200, cv=5, iid=False, verbose=1, n_jobs=32)

  logger.info("Starting random search...")
  t1 = datetime.datetime.now()
  random_search.fit(x_train, y_train)
  dt = datetime.datetime.now() - t1
  params_file = args.workdir/f'{task}_best_params.json'
  logger.info(f"Random search completed. Took {dt.days} days, {dt.seconds//3600} hours, and {(dt.seconds//60)%60} minutes. Writing best params to {params_file}")
  json.dump(random_search.best_params_, params_file.open('w'))
