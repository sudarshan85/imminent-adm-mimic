#!/usr/bin/env python

import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from args import args
from utils.splits import set_two_splits

def run(task, ori_df, threshold):
  preds = []
  targs = []
  probs = []
  print(f"Running for task: {task}")

  seeds = list(range(args.start_seed, args.start_seed + 100))

  for seed in tqdm(seeds, desc=f'{task} Runs'):
    df = set_two_splits(ori_df.copy(), 'test', seed=seed)
    vectorizer = TfidfVectorizer(min_df=args.min_freq, analyzer=str.split, ngram_range=(2,2))

    x_train = vectorizer.fit_transform(df.loc[(df['split'] == 'train')]['scispacy_note'])
    x_test = vectorizer.transform(df.loc[(df['split'] == 'test')]['scispacy_note'])

    y_train = df.loc[(df['split'] == 'train')][f'{task}_label'].to_numpy()
    y_test = df.loc[(df['split'] == 'test')][f'{task}_label'].to_numpy()
    targs.append(y_test)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)
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
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)

  imminent_df = ori_df.loc[(ori_df['imminent_label'] != -1)][['scispacy_note', 'imminent_label']].reset_index()
  discharge_df = ori_df[['scispacy_note', 'discharge_label']].reset_index()

  run('imminent', imminent_df, args.imminent_threshold)  
  run('discharge', discharge_df, args.discharge_threshold)