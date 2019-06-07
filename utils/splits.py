#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split

def test(a,b,cmp,cname=None):
  if cname is None: cname=cmp.__name__
  assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return np.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)

def set_all_splits(df, val_pct, test_pct=0.0, seed=None):
  new_test_pct = np.around(test_pct / (val_pct + test_pct), 2)
  train_pct = 1 - (val_pct + test_pct)
  train_idxs, inter = train_test_split(np.arange(len(df)), test_size=(val_pct + test_pct), random_state=seed)
  val_idxs, test_idxs = train_test_split(inter, test_size=new_test_pct, random_state=seed)

  df['split'] = None
  df.iloc[train_idxs, df.columns.get_loc('split')] = 'train'
  df.iloc[val_idxs, df.columns.get_loc('split')] = 'val'
  df.iloc[test_idxs, df.columns.get_loc('split')] = 'test'

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), train_pct)
  test_near(round(len(df[df['split'] == 'val'])/len(df), 2), val_pct)
  test_near(round(len(df[df['split'] == 'test'])/len(df), 2), test_pct)

  return df

def set_splits_with_sample(df, val_pct, test_pct=0.0, sample_pct=0.0, seed=None):
  new_test_pct = np.around(test_pct / (val_pct + test_pct), 2)
  train_pct = 1 - (val_pct + test_pct)
  train_idxs, inter = train_test_split(np.arange(len(df)), test_size=(val_pct + test_pct), random_state=seed)
  val_idxs, test_idxs = train_test_split(inter, test_size=new_test_pct, random_state=seed)

  df['split'] = None
  df.iloc[train_idxs, df.columns.get_loc('split')] = 'train'
  df.iloc[val_idxs, df.columns.get_loc('split')] = 'val'
  df.iloc[test_idxs, df.columns.get_loc('split')] = 'test'

  if sample_pct > 0.0:
    df['is_sample'] = False
    _, sample_idxs = train_test_split(train_idxs, test_size=sample_pct)
    df.iloc[sample_idxs, df.columns.get_loc('is_sample')] = True
    test_near(round(len(df[df['is_sample']])/len(df), 2), round(sample_pct * train_pct, 2))

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), train_pct)
  test_near(round(len(df[df['split'] == 'val'])/len(df), 2), val_pct)
  test_near(round(len(df[df['split'] == 'test'])/len(df), 2), test_pct)

  return df

def set_two_splits(df, name, pct=0.15, seed=None):
  df['split'] = 'train'
  _, val_idxs = train_test_split(np.arange(len(df)), test_size=pct, random_state=seed)
  df.loc[val_idxs, 'split'] = name

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), 1-pct)
  test_near(round(len(df[df['split'] == name])/len(df), 2), pct)

  return df

def set_bool_split(df, pct=0.15, seed=None):
  df['is_valid'] = False
  _, val_idxs = train_test_split(np.arange(len(df)), test_size=pct, random_state=seed)
  df.loc[val_idxs, 'is_valid'] = True

  test_near(round(len(df[df['is_valid'] == False])/len(df), 2), 1-pct)
  test_near(round(len(df[df['is_valid'] == True])/len(df), 2), pct)

  return df
