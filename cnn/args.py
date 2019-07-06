#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
glove_path = Path('../pretrained/glove')
fasttext_path = Path('../pretrained/fasttext')
workdir = Path('../data/workdir/cnn')

# all pretrained embedding paths
glove_50 = glove_path/'glove.6B.50d.txt'
glove_100 = glove_path/'glove.6B.100d.txt'
glove_200 = glove_path/'glove.6B.200d.txt'
glove_300 = glove_path/'glove.6B.300d.txt'
glove_mimic = glove_path/'glove.mimic.300d.txt'

ft_wiki_subword = fasttext_path/'wiki-news-300d-1M-subword.txt'
ft_wiki = fasttext_path/'wiki-news-300d-1M.txt'
ft_crawl_subword = fasttext_path/'crawl-300d-2M-subword.txt'
ft_crawl = fasttext_path/'crawl-300d-2M.txt'
ft_mimic = fasttext_path/'mimic-300d.txt'

args = Namespace(
  workdir=workdir,
  modeldir=workdir/'models',  
  dataset_csv=path/'proc_dataset.csv',
  cols=['hadm_id', 'imminent_adm_label', 'prolonged_stay_label', 'processed_note', 'charttime', 'intime'],
  imminent_adm_cols=['hadm_id', 'processed_note', 'imminent_adm_label'],
  prolonged_stay_cols=['hadm_id', 'processed_note', 'prolonged_stay_label'],
  dates=['charttime', 'intime'],
  batch_size=128,
  min_freq=3,
  hidden_dim=100,
  dropout_p=0.1,
  emb_dropout=0.1,
  n_channels=100,
  lr=1e-3,
  wd=1e-3,
  n_epochs=10,
  checkpointer_save_total=1,
  emb_path=glove_50,
  emb_sz=50,
  checkpointer_prefix='glove50_cnn',
  device='cuda:3',
  checkpointer_name='epoch',
  checkpointer_save_every=5,
  early_stop_patience=10,
  ia_thresh=0.52,
  ps_thresh=0.33,
  start_seed=127,
)
