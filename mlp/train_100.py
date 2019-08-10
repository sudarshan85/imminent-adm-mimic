#!/usr/bin/env python

import datetime
import logging
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

import pickle
import lightgbm
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from torch import optim

from skorch import NeuralNetBinaryClassifier
from skorch.toy import MLPModule
from skorch.dataset import CVSplit
from skorch.callbacks import *

from utils.splits import set_group_splits
from args import args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def run_100(task, task_df, args, threshold):
  reduce_lr = LRScheduler(
    policy='ReduceLROnPlateau',
    mode='min',
    factor=0.5,
    patience=1,
  )

  seeds = list(range(args.start_seed, args.start_seed + 100))
  for seed in tqdm(seeds, desc=f'{task} Runs'):
    logger.info(f"Spliting with seed {seed}")
    checkpoint = Checkpoint(
      dirname=args.modeldir/f'{task}_seed_{seed}',
    )
    df = set_group_splits(task_df.copy(), group_col='hadm_id', seed=seed)
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), binary=True, max_features=60_000)

    x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['processed_note']).astype(np.float32)
    x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['processed_note']).astype(np.float32)

    x_train = np.asarray(x_train.todense())
    x_test = np.asarray(x_test.todense())
    vocab_sz = len(vectorizer.vocabulary_)

    y_train = df.loc[(df['split'] == 'train')][f'{task}_label'].to_numpy()
    y_test = df.loc[(df['split'] == 'test')][f'{task}_label'].to_numpy()

    clf = MLPModule(input_units=vocab_sz, output_units=1, hidden_units=args.hidden_dim, num_hidden=1, dropout=args.dropout_p, squeeze_output=True)

    net = NeuralNetBinaryClassifier(
      clf,
      max_epochs=args.max_epochs,
      lr=args.lr,
      device=args.device,
      optimizer=optim.Adam,
      optimizer__weight_decay=args.wd,
      batch_size=args.batch_size,
      verbose=1,
      callbacks=[EarlyStopping, ProgressBar, checkpoint, reduce_lr],
      train_split=CVSplit(cv=0.15, stratified=True),
      iterator_train__shuffle=True,
      threshold=threshold,
    )
    net.set_params(callbacks__valid_acc=None)
    net.fit(x_train, y_train.astype(np.float32))

if __name__=='__main__':
  if len(sys.argv) != 2:
    logger.error(f"Usage: {sys.argv[0]} task_name (ia|ps)")
    sys.exit(1)

  task = sys.argv[1]
  if task not in ['ia', 'ps']:
    logger.error("Task values are either ia (imminent admission) or ps (prolonged stay)")
    sys.exit(1)

  args.modeldir = args.workdir/'models'
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols, parse_dates=args.dates)
  if task == 'ia':
    task_df = ori_df.loc[(ori_df['imminent_adm_label'] != -1)][args.imminent_adm_cols].reset_index(drop=True)
    prefix = 'imminent_adm'
    threshold = args.ia_thresh
  if task == 'ps':
    task_df = ps_df = ori_df.loc[(ori_df['chartinterval'] != 0)][args.prolonged_stay_cols].reset_index(drop=True)
    prefix = 'prolonged_stay'
    threshold = args.ps_thresh

  logger.info(f"Running 100 seed test run for task {task}")
  t1 = datetime.datetime.now()
  run_100(prefix, task_df, args, threshold)
  dt = datetime.datetime.now() - t1
  logger.info(f"100 seed test run completed. Took {dt.days} days, {dt.seconds//3600} hours, and {(dt.seconds//60)%60} minutes")
