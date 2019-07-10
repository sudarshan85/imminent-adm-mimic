#!/usr/bin/env python

import warnings
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from typing import List
from scipy import interp
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

def get_wordcloud(feature_names, scores, n_words='all'):
  if n_words == 'all':
    n_words = len(feature_names)

  p = re.compile('^[a-z\s]+$')
  neg_dict, pos_dict = {}, {}
  for word, score in zip(feature_names, scores):
    word = word.lower()
    if len(word) > 7 and word not in STOP_WORDS:
      if p.match(word):      
        neg_dict[word] = 1 - score
        pos_dict[word] = score

  neg_cloud = WordCloud(width=400, height=400, background_color='white', max_words=n_words, max_font_size=40, relative_scaling=0.5).generate_from_frequencies(neg_dict)
  pos_cloud = WordCloud(width=400, height=400, background_color='white', max_words=n_words, max_font_size=60, relative_scaling=0.5).generate_from_frequencies(pos_dict)  
    
  return neg_cloud, pos_cloud

def print_top_words(feature_names: List[str], probs: np.ndarray, N: int):
  words = sorted(zip(probs, feature_names), reverse=True)
  pos = words[:N]
  neg = words[:-(N + 1):-1]

  print("Words associated with imminent threat: ")
  for feat in pos:
    print(np.round(feat[0], 2), feat[1])

  print("***********************************************")
  print("Words associated with not imminent threat: ")
  for feat in neg:
    print(np.round(feat[0], 2), feat[1])

def plot_prob(ax, df, threshold, starting_day, ending_day, interval_hours, is_agg=False, is_log=False):
  if starting_day > 0:
    warnings.warn(f"starting_day ({starting_day}) must be negative. Converting it to negative")
    starting_day = -starting_day

  if ending_day > 0:
    warnings.warn(f"ending_day ({ending_day}) must be negative. Converting it to negative")
    ending_day = -ending_day

  if ending_day < starting_day:
    warnings.warn(f"starting_day ({starting_day}) must be less than ending_day ({ending_day}). Swapping values.")
    starting_day, ending_day = ending_day, starting_day

  high = pd.to_timedelta(ending_day, unit='d')
  low = pd.to_timedelta(starting_day, unit='d')  
  plot_data = df.loc[(df['relative_charttime'] > low) & (df['relative_charttime'] < high)][['relative_charttime', 'prob']].copy()
  plot_data['interval'] = ((plot_data['relative_charttime'].apply(lambda curr_time: int((curr_time - df['relative_charttime'].max())/pd.to_timedelta(interval_hours, unit='h')))))/2

  if is_agg:
    plot_data = plot_data[['interval', 'prob']].groupby(['interval']).agg(lambda x: np.average(x, weights=plot_data.loc[x.index, 'prob']))

  plot_data.reset_index(inplace=True)
  if is_log:
    plot_data['interval'] = -np.log1p(-plot_data['interval'])

  ax.axhline(y=threshold, label=f'Threshold = {threshold}', linestyle='--', color='r')
  sns.lineplot(x='interval', y='prob', data=plot_data, ax=ax)
  # ax.set_xlabel(f'Time to ICU (days)')
  # ax.set_ylabel('Probability')
  ax.set_xlabel('')
  ax.set_ylabel('')
  ax.legend(loc='upper left')
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

def plot_confusion_matrix(ax, cm, classes, normalize=False, title=None, cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)

  # We want to show all ticks and label them with the respective list entries
  ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes,
      yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] >
          thresh else 'black')

def plot_roc(ax, y_true, prob):
  fpr, tpr, _ = roc_curve(y_true, prob)
  ax.set_ylabel('Sensitivity')
  ax.set_xlabel('1 - Specificity')
  ax.plot([0, 1], [0, 1], linestyle='--')
  ax.plot(fpr, tpr, marker='.')
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)

def plot_mean_roc(ax, y_trues, probs, is_individual=False):
  curve_color = 'navy'
  if is_individual:
    curve_color  = 'white'

  tprs = []
  base_fpr = np.linspace(0, 1, len(y_trues))

  for i, (y_test, pos_prob) in enumerate(zip(y_trues, probs)):
    fpr, tpr, _ = roc_curve(y_test, pos_prob)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

  tprs = np.array(tprs)
  mean_tprs = tprs.mean(axis=0)
  std = tprs.std(axis=0)

  tprs_upper = np.minimum(mean_tprs + std, 1)
  tprs_lower = mean_tprs - std

  ax.plot(base_fpr, mean_tprs, color=curve_color, marker='.')
  if is_individual:
    for i, (y_test, pos_prob) in enumerate(zip(y_trues, probs)):
      fpr, tpr, _ = roc_curve(y_test, pos_prob)
      ax.plot(fpr, tpr, color='blue', alpha=0.15)
      ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

  ax.plot([0, 1], [0, 1], color='silver', linestyle=':')
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
  ax.set_ylabel('Sensitivity')
  ax.set_xlabel('1 - Specificity')
  

def plot_auprc(ax, y_true, probs):
  ap = average_precision_score(y_true, probs)
  precision, recall, _ = precision_recall_curve(y_true, probs)
  auprc = auc(recall, precision)

  ax.set_xlabel("Sensitivity")
  ax.set_ylabel("PPV")
  # ax.set_title("Precision-Recall Curve")
  ax.plot([0, 1], [0.5, 0.5], linestyle='--')
  ax.plot(recall, precision, marker='.')

  return ap, auprc

def plot_thresh_range(ax, y_true, prob, lower=0, upper=1, n_vals=5):
  metrics = np.zeros((4, n_vals))
  thresh_range = np.round(np.linspace(lower, upper, n_vals), 2)

  for i, thresh in enumerate(thresh_range):
    y_pred = (prob > thresh).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    metrics[0][i] = np.round(tp/(tp+fn), 3)
    metrics[1][i] = np.round(tn/(tn+fp), 3)
    metrics[2][i] = np.round(tp/(tp+fp), 3)
    metrics[3][i] = np.round(tn/(tn+fn), 3)

  df = pd.DataFrame(metrics, index=['sensitivity', 'specificity', 'ppv', 'npv'], columns=thresh_range)
  df=df.stack().reset_index()
  df.columns = ['Metric','Threshold','Value']
  ax = sns.pointplot(x='Threshold', y='Value', hue='Metric',data=df)
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
  ax.legend(loc='upper right')

def plot_youden(ax, y_true, prob, lower=0, upper=1, n_vals=5):
  youden_idxs = np.zeros(n_vals)
  thresh_range = np.round(np.linspace(lower, upper, n_vals), 2)
  
  for i, thresh in enumerate(thresh_range):
    y_pred = (prob > thresh).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    youden_idxs[i] = tp/(tp+fn) + tn/(tn+fp)
    
  youden_idxs = youden_idxs.reshape(1,-1)
  df = pd.DataFrame(youden_idxs, index=['youden_idx'], columns=thresh_range)
  df=df.stack().reset_index()
  df.columns = ['Metric','threshold','youden_idx']
  ax = sns.pointplot(x='threshold', y='youden_idx',data=df)
  ax.set_xlabel('Threshold')
  ax.set_ylabel('Youden Index')
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)  
