#!/usr/bin/env python

import datetime
import pickle
import logging
import sys
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
from collections import OrderedDict
from functools import partial

from torch import nn
from torch import optim

from ignite.metrics import Loss

from args import args
from utils.embeddings import PretrainedEmbeddings
from utils.splits import set_all_splits, set_two_splits

from cnn_classifier.dataset import NoteDataset
from cnn_classifier.model import NoteClassifier
from cnn_classifier.containers import ModelContainer, DataContainer
from cnn_classifier.trainer import IgniteTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def get_sample(df, sample_pct=0.01, with_val=True, seed=None):
  logger.debug("Grabbing sample")
  train = df.loc[(df['split']) == 'train'].sample(frac=sample_pct, random_state=seed)
  train.reset_index(inplace=True, drop=True)
  # val = df.loc[(df['split']) == 'val'].sample(frac=sample_pct, random_state=seed)
  # val.reset_index(inplace=True, drop=True)
  test = df.loc[(df['split']) == 'test'].sample(frac=sample_pct, random_state=seed)
  test.reset_index(inplace=True, drop=True)

  # return pd.concat([train, val, test], axis=0)
  return pd.concat([train, test], axis=0)

def convert_probs(output, thresh):
  y_pred, y = output
  y_pred = (torch.sigmoid(y_pred) > thresh).long()
  return y_pred, y

def predict_proba(clf, x_test):
  return torch.sigmoid(clf(x_test)).detach().numpy()

if __name__=='__main__':

  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} #(1-4)")
    sys.exit(-1)

  partition = int(sys.argv[1])
  if partition not in [1, 2, 3, 4]:
    print(f"Usage: {sys.argv[0]} #(1-4)")
    sys.exit(-1)

  args.device = f'cuda:{partition-1}'
  l = list(range(args.start_seed, args.start_seed+100))
  seeds = [l[i:i + 25] for i in range(0, len(l), 25)][partition-1]
  print(args.device, seeds)
  sys.exit(-1)

  preds = []
  targs = []
  probs = []

  logger.info("Starting run")

  logger.debug("Loading data...")
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)
  t1 = datetime.datetime.now()
  prefix = args.checkpointer_prefix

  for seed in seeds:
    args.checkpointer_prefix = prefix + '_seed_' + str(seed)
    logger.info(f"Splitting data with seed: {seed}")
    df = set_two_splits(ori_df.copy(), 'test', seed=seed)
    # df = get_sample(set_two_splits(ori_df.copy(), 'test', seed=seed))
    dc = DataContainer(df, NoteDataset, args.workdir, bs=args.batch_size, with_test=True,
        min_freq=args.min_freq, create_vec=True, weighted_sampling=True)

    pe = PretrainedEmbeddings.from_file(args.emb_path)
    pe.make_custom_embeddings(dc.get_vocab_tokens())

    logger.debug("Creating model...")
    classifier = NoteClassifier(args.emb_sz, dc.get_vocab_size(), args.n_channels, args.hidden_dim,
        dc.n_classes, dropout_p=args.dropout_p, emb_dropout=args.emb_dropout, pretrained=pe.custom_embeddings)

    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()

    mc = ModelContainer(classifier, loss_fn, optimizer, reduce_lr)
    metrics = OrderedDict({'loss': Loss(loss_fn)})

    ig = IgniteTrainer(mc, dc, args, metrics, log_training=False, early_stop=False, verbose=True)
    ig.run()

    # load the latest model
    model_file = args.checkpointer_prefix + '_' + args.checkpointer_name + '_' + str(args.n_epochs) + '.pth'
    logger.info(f"Loading model from {model_file}")

    classifier.load_state_dict(torch.load(args.workdir/f'models/{model_file}'))
    x_test, targ = next(iter(dc.test_dl))

    x_test = x_test.to('cpu')
    targ = targ.to('cpu')
    classifier = classifier.to('cpu')

    prob = predict_proba(classifier, x_test)
    pred = (prob > args.bc_threshold).astype(np.int64)
    targs.append(targ)
    preds.append(pred)
    probs.append(prob)
    torch.cuda.empty_cache()

  dt = datetime.datetime.now() - t1
  logger.info(f"{len(seeds)} runs completed. Took {dt.seconds//3600} hours and {(dt.seconds//60)%60} minutes.")
  preds_file = args.workdir/f'preds_{partition}.pkl'
  logger.info(f"Writing predictions to {preds_file}.")

  with open(preds_file, 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(preds, f)
    pickle.dump(probs, f)
