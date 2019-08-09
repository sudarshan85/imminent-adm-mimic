#!/usr/bin/env python

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from functools import partial
from typing import List
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy import stats

def _mean_confidence_interval(data, conf=0.95, decimal=3):
  assert(conf > 0 and conf < 1), f"Confidence interval must be within (0, 1). It is {conf}"
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), stats.sem(a)
  h = se * stats.t.ppf((1 + conf) / 2., n-1)
  return np.round(m, 3), np.round(m-h, decimal), np.round(m+h, decimal)

class BinaryAvgMetrics(object):
  def __init__(self, targets: List[int], predictions: List[int], probs: List[float], decimal=3) -> None:
    assert (len(targets) == len(predictions) == len(probs)), f"Target list (length = {len(targets)}), predictions list (length = {len(predictions)}) and probabilities list (length = {len(probs)}) must all be of the same length!))"
    self.targs = targets
    self.n_runs = len(self.targs)
    self.preds = predictions
    self.probs = probs
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
  
  @property
  def prevalence_avg(self):
    return np.round(((self.fns + self.tps) / (self.tns + self.fps + self.fns + self.tps)).mean(), self.decimal)

  def sensitivities(self):
    return self.tps / (self.tps + self.fns)
  
  def sensitivity_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    if conf is not None:
      return _mean_confidence_interval(se, conf)

    return np.round(se.mean(), self.decimal,)

  def specificities(self):
    return self.tns / (self.tns + self.fps)
  
  def specificity_avg(self, conf=None):
    sp = (self.tns / (self.tns + self.fps))
    if conf is not None:
      return _mean_confidence_interval(sp, conf)

    return np.round(sp.mean(), self.decimal)

  def ppvs(self):
    return self.tps / (self.tps + self.fps)
  
  def ppv_avg(self, conf=None):
    ppv = (self.tps / (self.tps + self.fps))
    if conf is not None:
      return _mean_confidence_interval(ppv, conf)

    return np.round(ppv.mean(), self.decimal)  

  def npvs(self):
    return self.tns / (self.tns + self.fns)
  
  def npv_avg(self, conf=None):
    npv = (self.tns / (self.tns + self.fns))
    if conf is not None:
      return _mean_confidence_interval(npv, conf)

    return np.round(npv.mean(), self.decimal)
  
  def f1s(self):
    return (2 * self.sensitivities() * self.ppvs()) / (self.sensitivities() + self.ppvs())

  def f1_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    ppv = (self.tps / (self.tps + self.fps))
    f1 = (2 * se * ppv) / (se + ppv)
    if conf is not None:
      return _mean_confidence_interval(f1, conf)

    return np.round(f1.mean(), self.decimal)

  def aurocs(self):
    return np.array([roc_auc_score(targ, prob) for targ, prob in zip(self.targs, self.probs)])

  def auroc_avg(self, conf=None):
    auroc = np.array([roc_auc_score(targ, prob) for targ, prob in zip(self.targs, self.probs)])
    if conf is not None:
      return _mean_confidence_interval(auroc, conf)

    return np.round(auroc.mean(), self.decimal)

  def get_avg_metrics(self, conf=None, defn=False):
    definitions = {
      'sensitivity': "When it's ACTUALLY YES, how often does it PREDICT YES?",
      'specificity': "When it's ACTUALLY NO, how often does it PREDICT NO?",
      'ppv': "When it PREDICTS YES, how often is it correct?",
      'auroc': "Indicates how well the model is capable of distinguishing between classes",
      'npv': "When it PREDICTS NO, how often is it correct?",
      'f1': "Harmonic mean of sensitivity and ppv",
    }
    if conf is None:
      metrics = {
        'sensitivity': [self.sensitivity_avg() * 100],
        'specificity': [self.specificity_avg() * 100],
        'ppv': [self.ppv_avg() * 100],
        'auroc': [self.auroc_avg() * 100],
        'npv': [self.npv_avg() * 100],
        'f1': [self.f1_avg() * 100],
      }

      if defn:
        for metric, value in metrics.items():
          value.append(definitions[metric])
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Value', 'Definition'])
      else:
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Value'])

      return d

    else:
      metrics = {
        'sensitivity': [*[value * 100 for value in self.sensitivity_avg(conf)]],        
        'specificity': [*[value * 100 for value in self.specificity_avg(conf)]],
        'ppv': [*[value * 100 for value in self.ppv_avg(conf)]],
        'auroc': [*[value * 100 for value in self.auroc_avg(conf)]],   
        'npv': [*[value * 100 for value in self.npv_avg(conf)]],
        'f1': [*[value * 100 for value in self.f1_avg(conf)]],        
      }

      if defn:
        for metric, value in metrics.items():
          value.append(definitions[metric])
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Mean', 'Lower', 'Upper', 'Definition'])
      else:
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Mean', 'Lower', 'Upper'])

      return d
  
  def __repr__(self):
    s = f"Number of Runs: {self.n_runs}\n"
    return s
  
  def __len__(self):
    return len(self.targs)

def get_best_model(bam: BinaryAvgMetrics, fnames: List[str]):
  best_se, best_se_model = 0, None
  best_sp, best_sp_model = 0, None
  best_ppv, best_ppv_model = 0, None
  best_auroc, best_auroc_model = 0, None
  best_npv, best_npv_model = 0, None
  best_f1, best_f1_model = 0, None

  for i in range(bam.n_runs):
    se = bam.tps[i] / (bam.tps[i] + bam.fns[i])
    sp = bam.tns[i] / (bam.tns[i] + bam.fps[i])
    ppv = bam.tps[i] / (bam.tps[i] + bam.fps[i])
    npv = bam.tns[i] / (bam.tns[i] + bam.fns[i])
    f1 = (2 * se * ppv) / (se + ppv)

    if best_se < se:
      best_se = se
      best_se_model = fnames[i]    
    if best_sp < sp:
      best_sp = sp
      best_sp_model = fnames[i]          
    if best_ppv < ppv:
      best_ppv = ppv
      best_ppv_model = fnames[i]    
    if best_npv < npv:
      best_npv = npv
      best_npv_model = fnames[i]  
    if best_f1 < f1:
      best_f1 = f1
      best_f1_model = fnames[i]    

  for i, (targ, prob) in enumerate(zip(bam.targs, bam.probs)):
    auroc = roc_auc_score(targ, prob)
    if best_auroc < auroc:
      best_auroc = auroc
      best_auroc_model = fnames[i]

  d = {
    'sensitivity': [best_se, best_se_model],
    'specificity': [best_sp, best_sp_model],
    'ppv': [best_ppv, best_ppv_model],
    'auroc': [best_auroc, best_auroc_model],
    'npv': [best_npv, best_npv_model],
    'f1': [best_f1, best_f1_model],
  }

  return pd.DataFrame(d.values(), index=d.keys(), columns=['Value', 'Model File'])
