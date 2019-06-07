#!/usr/bin/env python

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from functools import partial
from typing import List
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy import stats

def mean_confidence_interval(data, conf=0.95, decimal=3):
  assert(conf > 0 and conf < 1), f"Confidence interval must be within (0, 1). It is {conf}"
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), stats.sem(a)
  h = se * stats.t.ppf((1 + conf) / 2., n-1)
  return np.round(m-h, decimal), np.round(m, 3), np.round(m+h, decimal)

class BinaryAvgMetrics(object):
  def __init__(self, targets: List[int], predictions: List[int], probs: List[float], names: List[str], decimal=3) -> None:
    assert (len(targets) == len(predictions) == len(probs)), f"Target list (length = {len(targets)}), predictions list (length = {len(predictions)}) and probabilities list (length = {len(probs)}) must all be of the same length!))"
    self.targs = targets
    self.preds = predictions
    self.probs = probs
    self.names = names
    self.decimal = 3
    
    self.cms = np.zeros((len(self.targs), 2, 2), dtype=np.int64)

    for i, (targ, pred) in enumerate(zip(self.targs, self.preds)):
      self.cms[i] = confusion_matrix(targ, pred)  

  @property
  def tns(self):
    return self.cms[:, 0, 0]
  
  @property
  def fps(self):
    return self.cms[:, 0, 1]
  
  @property
  def fns(self):
    return self.cms[:, 1, 0]
  
  @property
  def tps(self):
    return self.cms[:, 1, 1]
  
  @property
  def cm_avg(self):
    return np.ceil(np.array([[self.tns.mean(), self.fps.mean()], [self.fns.mean(), self.tps.mean()]])).astype(np.int64)
  
  def prevalence_avg(self):
    return np.round(((self.fns + self.tps) / (self.tns + self.fps + self.fns + self.tps)).mean(), self.decimal)
  
  def sensitivity_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    if conf is not None:
      return mean_confidence_interval(se, conf)

    return np.round(se.mean(), self.decimal,)
  
  def specificity_avg(self, conf=None):
    sp = (self.tns / (self.tns + self.fps))
    if conf is not None:
      return mean_confidence_interval(sp, conf)

    return np.round(sp.mean(), self.decimal)
  
  def ppv_avg(self, conf=None):
    ppv = (self.tps / (self.tps + self.fps))
    if conf is not None:
      return mean_confidence_interval(ppv, conf)

    return np.round(ppv.mean(), self.decimal)  
  
  def npv_avg(self, conf=None):
    npv = (self.tns / (self.tns + self.fns))
    if conf is not None:
      return mean_confidence_interval(npv, conf)

    return np.round(npv.mean(), self.decimal)
  
  @property
  def f1_avg(self):
    return np.round(((2 * self.ppv_avg() * self.sensitivity_avg()) / (self.ppv_avg() + self.sensitivity_avg())).mean(), self.decimal)

  @property
  def auroc_avg(self):
    return np.round(np.mean([roc_auc_score(targ, prob) for targ, prob in zip(self.targs, self.probs)]), self.decimal)
  
  def get_avg_metrics(self, conf=None):
    if conf is None:
      d = {
        'sensitivity': [self.sensitivity_avg(), "When it's ACTUALLY YES, how often does it PREDICT YES?"],
        'specificity': [self.specificity_avg(), "When it's ACTUALLY NO, how often does it PREDICT NO?"],
        'ppv': [self.ppv_avg(), "When it PREDICTS YES, how often is it correct?"],
        'auroc': [self.auroc_avg, "Indicates how well the model is capable of distinguishing between classes"],
        'npv': [self.npv_avg(), "When it PREDICTS NO, how often is it correct?"],
        'f1': [self.f1_avg, "Harmonic mean of sensitivity and ppv"],
      }
    
      return pd.DataFrame(d.values(), index=d.keys(), columns=['Value', 'Definition'])
    else:
      d = {
        'sensitivity': [*self.sensitivity_avg(conf), conf, "When it's ACTUALLY YES, how often does it PREDICT YES?"],
        'specificity': [*self.specificity_avg(conf), conf, "When it's ACTUALLY NO, how often does it PREDICT NO?"],
        'ppv': [*self.ppv_avg(conf), conf, "When it PREDICTS YES, how often is it correct?"],
        'npv': [*self.npv_avg(conf), conf, "When it PREDICTS NO, how often is it correct?"],
      }

      return pd.DataFrame(d.values(), index=d.keys(), columns=['Lower', 'Mean', 'Upper', 'Confidence', 'Definition'])
  
  def __repr__(self):
    s = f"Number of Runs: {len(self.targs)}\n"
    s += f"Average Prevalence of positive class: {self.prevalence_avg()}"
    return s
