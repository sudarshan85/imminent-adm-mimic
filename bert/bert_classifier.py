#!/usr/bin/env python

import pdb
import logging
import random
import pickle
import numpy as np
import datetime
import pandas as pd
import torch
import sys
sys.path.append('../')

from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
from tqdm import tqdm, trange
from sklearn.metrics import recall_score, roc_auc_score, precision_score

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from args import args
from utils.splits import set_two_splits

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

@dataclass
class InputExample(object):
  """
    A single training/test example for simple sequence classification.
  """
  eid: str
  text: str
  label: int

@dataclass
class InputFeatures(object):
  """
    A single set of features representing the data.
  """
  input_ids: List[int]
  input_mask: List[int]
  segment_ids: List[int]
  label_id: List[Optional[int]]

def read_df(df: pd.DataFrame, txt_col, label_col, set_type='train') -> List[InputExample]:
  """
    Function to read a dataframe with text and label and create a list
    of InputExample
  """
  logger.debug(f"Reading column text column {txt_col} and label column {label_col}")
  examples = []  
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    eid = f'{set_type}-{i}'
    text = row[txt_col]
    label = row[label_col]
    examples.append(InputExample(eid=eid, text=text, label=label))

  return examples

def get_sample(df: pd.DataFrame, sample_pct: float=0.01, val_test: Optional[str]=None, seed: Optional[int]=None):
  train = df.loc[(df['split']) == 'train'].sample(frac=sample_pct, random_state=seed)
  train.reset_index(inplace=True, drop=True)

  if val_test is not None:
    val = df.loc[(df['split']) == val_test].sample(frac=sample_pct, random_state=seed)
    val.reset_index(inplace=True, drop=True)
    return pd.concat([train, val], axis=0) 

  return train

def convert_examples_to_features(examples: List[InputExample], label_list: List[Union[int, str]], max_seq_len: int, tokenizer: BertTokenizer, is_pred=False) -> List[InputFeatures]:
  """
    Loads a data file into a list
  """
  features = []
  n_trunc = 0
  for example in tqdm(examples):
    tokens = tokenizer.tokenize(example.text)
    
    # Account for [CLS], [SEP] with -2
    if len(tokens) > max_seq_len - 2:
      logger.debug(f"Sample with ID '{example.eid}' has sequence length {len(tokens)} greater than max_seq_len ({max_seq_len}). Truncating sequence.")      
      tokens = tokens[:(max_seq_len - 2)]
      n_trunc += 1

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are 
    # attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    if is_pred:
      label_id = None
    else:
      label_id = example.label
    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))

  logger.warning(f"{(n_trunc/len(examples))*100:0.1f}% ({n_trunc}) of total examples have sequence length longer than max_seq_len ({max_seq_len})")
  return features

def compute_metrics(preds, labels):
  assert len(preds) == len(labels)
  return {
    'sensitivity': recall_score(labels, preds),
    'ppv': precision_score(labels, preds),
    'auroc': roc_auc_score(labels, preds),
  }

def set_global_seed(seed=None):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

def train(train_dataloader, num_train_optimization_steps):

  # Prepare model
  model = BertForSequenceClassification.from_pretrained(args.bert_dir, num_labels=args.num_labels)
  model.to(args.device)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
  optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_prop, t_total=num_train_optimization_steps)

  loss_fct = nn.BCEWithLogitsLoss()
  avg_epoch_loss = 0
  model.train()
  for _ in trange(int(args.n_epochs), desc="Epoch"):
    epoch_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
      batch = tuple(t.to(args.device) for t in batch)
      input_ids, input_mask, segment_ids, label_ids = batch

      logits = model(input_ids, segment_ids, input_mask, labels=None)
      loss = loss_fct(logits.view(-1), label_ids.float())

      if args.n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      loss.backward()
      epoch_loss += loss.item()
      nb_tr_steps += 1
      if (step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_epoch_loss += epoch_loss / nb_tr_steps

  model_to_save = model.module if hasattr(model, 'module') else model
  torch.save(model_to_save.state_dict(), args.modeldir/'pytorch_model.bin')

  return avg_epoch_loss / args.n_epochs

def evaluation(eval_dataloader):
  # Load a trained model and vocabulary that you have fine-tuned
  model = BertForSequenceClassification.from_pretrained(args.modeldir, num_labels=args.num_labels)
  model.to(args.device)
  loss_fct = nn.BCEWithLogitsLoss()

  model.eval()
  eval_loss = 0
  nb_eval_steps = 0
  preds = []
  probs = []

  for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(args.device)
    input_mask = input_mask.to(args.device)
    segment_ids = segment_ids.to(args.device)
    label_ids = label_ids.to(args.device)

    with torch.no_grad():
      logits = model(input_ids, segment_ids, input_mask, labels=None)

    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.float())  
    eval_loss += tmp_eval_loss.mean().item()      
    nb_eval_steps += 1

    prob = torch.sigmoid(logits).detach().cpu().numpy()
    if len(probs) == 0:
      probs.append(prob)
    else:
      probs[0] = np.append(
        probs[0], prob, axis=0)

    pred = (prob > args.bc_threshold).astype(np.int64)
    if len(preds) == 0:
      preds.append(pred)
    else:
      preds[0] = np.append(
        preds[0], pred, axis=0)

  eval_loss = eval_loss / nb_eval_steps
  probs = np.squeeze(probs[0])
  preds = np.squeeze(preds[0])

  return eval_loss, probs, preds

def main():
  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)
  # df = get_sample(set_two_splits(ori_df.copy(), name='test'), val_test='test', seed=seed)  
  # df = set_two_splits(ori_df.copy(), 'test')

  logger.info(f"device: {args.device} n_gpu: {args.n_gpu}")
  # seeds = list(range(args.start_seed, args.start_seed + 100))
  seeds = list(range(42, 44))

  if args.gradient_accumulation_steps < 1:
    raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")

  args.bs = args.bs // args.gradient_accumulation_steps
  t1 = datetime.datetime.now()
  tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.do_lower_case)
  
  preds, targs, probs = [], [], []
  for seed in seeds:
    # seed = 42
    set_global_seed(seed)
    logger.info(f"Splitting data with seed: {seed}")
    df = get_sample(set_two_splits(ori_df.copy(), name='test'), val_test='test', seed=seed)  
    # df = set_two_splits(ori_df.copy(), 'test')

    if args.do_train:
      train_examples = read_df(df.loc[(df['split'] == 'train')], 'note', 'class_label')
      train_features = convert_examples_to_features(train_examples, args.labels, args.max_seq_len, tokenizer)
      num_train_optimization_steps = int(len(train_examples) / args.bs /args.gradient_accumulation_steps) * args.n_epochs
      
      logger.info("***** Running training *****")
      logger.info("  Num examples = %d", len(train_examples))
      logger.info("  Batch size = %d", args.bs)
      logger.info("  Num steps = %d", num_train_optimization_steps)

      all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
      all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

      train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
      train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.bs, drop_last=True)
      loss = train(train_dataloader, num_train_optimization_steps)

      logger.info(f"Final average loss: {loss:0.3f}")

    if args.do_eval:
      eval_examples = read_df(df.loc[(df['split'] == 'test')], 'note', 'class_label', set_type='test')  
      eval_features = convert_examples_to_features(eval_examples, args.labels, args.max_seq_len, tokenizer)
      logger.info("***** Running evaluation *****")
      logger.info("  Num examples = %d", len(eval_examples))
      logger.info("  Batch size = %d", args.bs)
      all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
      all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

      eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

      # Run prediction for full data
      eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=args.bs)
      eval_loss, prob, pred = evaluation(eval_dataloader)
      assert(len(prob) == len(pred) == len(all_label_ids.numpy()))

      result = compute_metrics(pred, all_label_ids.numpy())
      logger.info("***** Eval results *****")
      logger.info(f"  Evaluation Loss: {eval_loss}")
      for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

      model_file = args.modeldir/f'pytorch_model.bin'
      model_file.rename(args.modeldir/f'bert_seed_{seed}.pth')

      # preds.append(pred)
      # probs.append(prob)
      # targs.append(all_label_ids.numpy())
      # torch.cuda.empty_cache()

  # dt = datetime.datetime.now() - t1
  # logger.info(f"{len(seeds)} runs completed. Took {dt.seconds//3600} hours and {(dt.seconds//60)%60} minutes.")
  # preds_file = args.workdir/f'preds.pkl'
  # logger.info(f"Writing predictions to {preds_file}.")

  # with open(preds_file, 'wb') as f:
  #   pickle.dump(targs, f)
  #   pickle.dump(preds, f)
  #   pickle.dump(probs, f)
        
if __name__ == "__main__":
  main()
