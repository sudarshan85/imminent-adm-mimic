#!/usr/bin/env python

import logging
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from args import args, ia_params, ps_params
from utils.splits import set_group_splits

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def run(task, ori_df, params, threshold):
  preds = []
  targs = []
  probs = []
  logger.info(f"Running for task: {task}")

  seeds = list(range(args.start_seed, args.start_seed + 100))

  for seed in tqdm(seeds, desc=f'{task} Runs'):
    df = set_group_splits(task_df.copy(), group_col='hadm_id', seed=seed)
    vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, ngram_range=(2,2))

    x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note'])
    x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note'])

    y_train = df.loc[(df['split'] == 'train')][f'{task}_label'].to_numpy()
    y_test = df.loc[(df['split'] == 'test')][f'{task}_label'].to_numpy()
    targs.append(y_test)

    clf = LogisticRegression(**params)
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
  if len(sys.argv) != 2:
    logger.error(f"Usage: {sys.argv[0]} task_name (ia|ps)")
    sys.exit(1)

  task = sys.argv[1]
  if task not in ['ia', 'ps']:
    logger.error("Task values are either ia (imminent admission) or ps (prolonged stay)")
    sys.exit(1)

  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  if task == 'ia':
    task_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
    prefix = 'imminent_adm'
    params = ia_params
    threshold = args.ia_thresh
  if task == 'ps':
    task_df = ps_df = ori_df.loc[(ori_df['chartinterval'] != 0)][args.prolonged_stay_cols].reset_index(drop=True)
    prefix = 'prolonged_stay'
    params = ps_params
    threshold = args.ps_thresh

  run(prefix, task_df, params, threshold)
