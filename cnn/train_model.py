#!/usr/bin/env python

import logging
import sys
sys.path.append('../')

import torch
import pandas as pd
from collections import OrderedDict
from functools import partial

from torch import nn
from torch import optim

from ignite.metrics import Loss
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from args import args
from utils.embeddings import PretrainedEmbeddings
from utils.splits import set_all_splits
from classifier.dataset import NoteDataset
from classifier.model import NoteClassifier
from classifier.containers import ModelContainer, DataContainer
from classifier.trainer import IgniteTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def get_sample(df, sample_pct=0.01, with_val=True, seed=None):
  logger.debug("Grabbing sample")
  train = df.loc[(df['split']) == 'train'].sample(frac=sample_pct, random_state=seed)  
  train.reset_index(inplace=True, drop=True)

  if with_val:
    val = df.loc[(df['split']) == 'val'].sample(frac=sample_pct, random_state=seed)
    val.reset_index(inplace=True, drop=True)
    return pd.concat([train, val], axis=0) 

  return train

if __name__=='__main__':
  logger.info("Loading data...")
  ori_df = pd.read_csv(args.dataset_csv)
  ori_df.drop(['note'], axis=1, inplace=True)

  seed = 42
  args.checkpointer_prefix = args.checkpointer_prefix + '_seed_' + str(seed)
  logger.info(f"Splitting data with seed: {seed}")
  df = set_all_splits(ori_df.copy(), 0.1, 0.1, seed=seed)
  sample_df = sample_df = get_sample(df)
  dc = DataContainer(sample_df, NoteDataset, args.workdir, bs=args.batch_size, with_test=False,
      min_freq=args.min_freq, load_vec=False, weighted_sampling=True)

  pe = PretrainedEmbeddings.from_file(args.emb_path)
  pe.make_custom_embeddings(dc.get_vocab_tokens())

  logger.info("Creating model...")
  classifier = NoteClassifier(args.emb_sz, dc.get_vocab_size(), args.n_channels, args.hidden_dim,
      dc.n_classes, dropout_p=args.dropout_p, emb_dropout=args.emb_dropout, pretrained=pe.custom_embeddings)

  optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
  reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=1)
  loss_fn = nn.BCEWithLogitsLoss()

  mc = ModelContainer(classifier, loss_fn, optimizer, reduce_lr)
  metrics = OrderedDict({'loss': Loss(loss_fn)})

  args.n_epochs = 2
  ig = IgniteTrainer(mc, dc, args, metrics, log_training=True, early_stop=True)
  best_model = ig.run()
  logger.info(f"Best Model Name: {best_model}")
