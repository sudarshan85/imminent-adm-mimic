#!/usr/bin/env python

import datetime
import logging
import sys

import warnings
warnings.filterwarnings("ignore")

import pickle
import lightgbm
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils.splits import set_group_splits
import lr.args
import rf.args
import gbm.args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def run_100(task, ori_df, clf_model, params, args, threshold):
  preds = []
  targs = []
  probs = []

  seeds = list(range(args.start_seed, args.start_seed + 100))
  for seed in tqdm(seeds, desc=f'{task} Runs'):
    df = set_group_splits(task_df.copy(), group_col='hadm_id', seed=seed)
    vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, ngram_range=(2,2))

    x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note'])
    x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note'])

    y_train = df.loc[(df['split'] == 'train')][f'{task}_label'].to_numpy()
    y_test = df.loc[(df['split'] == 'test')][f'{task}_label'].to_numpy()
    targs.append(y_test)

    clf = clf_model(**params)
    clf.fit(x_train, y_train)  
    pickle.dump(clf, open(args.modeldir/f'{task}_seed_{seed}.pkl', 'wb'))

    pos_prob = clf.predict_proba(x_test)[:, 1]
    probs.append(pos_prob)

    y_pred = (pos_prob > threshold).astype(np.int64)
    preds.append(y_pred)

  with open(args.workdir/f'{task}_preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(preds, f)
    pickle.dump(probs, f)

if __name__=='__main__':
  if len(sys.argv) != 3:
    logger.error(f"Usage: {sys.argv[0]} task_name (ia|ps) model_name (lr|rf|gbm)")
    sys.exit(1)

  task = sys.argv[1]
  if task not in ['ia', 'ps']:
    logger.error("Task values are either ia (imminent admission) or ps (prolonged stay)")
    sys.exit(1)

  clf_name = sys.argv[2]
  if clf_name not in ['lr', 'rf', 'gbm']:
    logger.error("Allowed models are lr (logistic regression), rf (random forest), or gbm (gradient boosting machines)")
    sys.exit(1)

  if clf_name == 'lr':
    clf_model = LogisticRegression
    args = lr.args.args
    ia_params = lr.args.ia_params
    ps_params = lr.args.ps_params
  elif clf_name == 'rf':
    clf_model = RandomForestClassifier
    args = rf.args.args
    ia_params = rf.args.ia_params
    ps_params = rf.args.ps_params
  else:
    clf_model = lightgbm.LGBMClassifier
    args = gbm.args.args
    ia_params = gbm.args.ia_params
    ps_params = gbm.args.ps_params

  args.dataset_csv =  Path('./data/proc_dataset.csv')
  args.workdir = Path(f'./data/workdir/{clf_name}')
  args.modeldir = args.workdir/'models'
  
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  if task == 'ia':
    task_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
    prefix = 'imminent_adm'
    params = ia_params
    threshold = args.ia_thresh
  if task == 'ps':
    task_df = ori_df[args.prolonged_stay_cols].copy()
    prefix = 'prolonged_stay'
    params = ps_params
    threshold = args.ps_thresh
  
  logger.info(args.workdir)
  logger.info(args.modeldir)
  logger.info(f"Running 100 seed test run for task {task} with model {clf_name}") 
  t1 = datetime.datetime.now()
  run_100(prefix, task_df, clf_model, params, args, threshold)
  dt = datetime.datetime.now() - t1
  logger.info(f"100 seed test run completed. Took {dt.seconds//3600} hours and {(dt.seconds//60)%60} minutes")