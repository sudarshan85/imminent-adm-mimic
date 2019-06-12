#!/usr/bin/env python

import pdb
import logging
import random
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../')

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


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
      sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
      Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
      specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

def read_df(df: pd.DataFrame, txt_col, label_col, set_type='train') -> List[InputExample]:
  """
    Function to read a dataframe with text and label and create a list
    of InputExample
  """
  logger.debug(f"Reading column text column {txt_col} and label column {label_col}")
  examples = []  
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    guid = f'{set_type}-{i}'
    text_a = row[txt_col]
    label = row[label_col]
    examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

  return examples

def get_sample(df, sample_pct=0.01, with_val=True, seed=None):
  train = df.loc[(df['split']) == 'train'].sample(frac=sample_pct, random_state=seed)
  train.reset_index(inplace=True, drop=True)

  if with_val:
    val = df.loc[(df['split']) == 'val'].sample(frac=sample_pct, random_state=seed)
    val.reset_index(inplace=True, drop=True)
    return pd.concat([train, val], axis=0) 

  return train  


def convert_examples_to_features(examples, label_list, max_seq_length,
                 tokenizer, output_mode):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {label : i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
      label_id = label_map[example.label]
    elif output_mode == "regression":
      label_id = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("tokens: %s" % " ".join(
          [str(x) for x in tokens]))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logger.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def compute_metrics(preds, labels):
  assert len(preds) == len(labels)
  return {
    'sensitivity': recall_score(labels, preds),
    'ppv': precision_score(labels, preds),
    'auroc': roc_auc_score(labels, preds),
  }

def set_global_seed(seed=None, n_gpu=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

def main():
  seed=42
  # n_gpu = torch.cuda.device_count()
  n_gpu = 0
  set_global_seed(seed, n_gpu)

  ori_df = pd.read_csv(args.dataset_csv, usecols=args.cols)
  df = get_sample(set_two_splits(ori_df.copy(), 'val'), seed=seed)

  logger.info(f"device: {args.device} n_gpu: {n_gpu}")

  if args.gradient_accumulation_steps < 1:
    raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")

  args.bs = args.bs // args.gradient_accumulation_steps
  logger.info(args.bs)

  label_list = ["0", "1"]
  num_labels = 1

  tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.do_lower_case)

  train_examples = None
  num_train_optimization_steps = None
  if args.do_train:
    train_examples = read_df(df.loc[(df['split'] == 'train')], 'note', 'class_label')
    num_train_optimization_steps = int(len(train_examples) / args.bs /args.gradient_accumulation_steps) * args.n_epochs
    logger.info(f"{len(train_examples)}, {num_train_optimization_steps}")


  # Prepare model
  # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
  # model = BertForSequenceClassification.from_pretrained(args.bert_model,
  #       cache_dir=cache_dir,
  #       num_labels=num_labels)
  # if args.fp16:
  #   model.half()
  # model.to(device)
  # if args.local_rank != -1:
  #   try:
  #     from apex.parallel import DistributedDataParallel as DDP
  #   except ImportError:
  #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

  #   model = DDP(model)
  # elif n_gpu > 1:
  #   model = torch.nn.DataParallel(model)

  # # Prepare optimizer
  # if args.do_train:
  #   param_optimizer = list(model.named_parameters())
  #   no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  #   optimizer_grouped_parameters = [
  #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
  #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  #     ]
  #   if args.fp16:
  #     try:
  #       from apex.optimizers import FP16_Optimizer
  #       from apex.optimizers import FusedAdam
  #     except ImportError:
  #       raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

  #     optimizer = FusedAdam(optimizer_grouped_parameters,
  #                 lr=args.learning_rate,
  #                 bias_correction=False,
  #                 max_grad_norm=1.0)
  #     if args.loss_scale == 0:
  #       optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
  #     else:
  #       optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
  #     warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
  #                        t_total=num_train_optimization_steps)

  #   else:
  #     optimizer = BertAdam(optimizer_grouped_parameters,
  #                lr=args.learning_rate,
  #                warmup=args.warmup_proportion,
  #                t_total=num_train_optimization_steps)

  # global_step = 0
  # nb_tr_steps = 0
  # tr_loss = 0
  # if args.do_train:
  #   train_features = convert_examples_to_features(
  #     train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
  #   logger.info("***** Running training *****")
  #   logger.info("  Num examples = %d", len(train_examples))
  #   logger.info("  Batch size = %d", args.train_batch_size)
  #   logger.info("  Num steps = %d", num_train_optimization_steps)
  #   all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
  #   all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
  #   all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

  #   if output_mode == "classification":
  #     all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
  #   elif output_mode == "regression":
  #     all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

  #   train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  #   if args.local_rank == -1:
  #     train_sampler = RandomSampler(train_data)
  #   else:
  #     train_sampler = DistributedSampler(train_data)
  #   train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

  #   model.train()
  #   for _ in trange(int(args.num_train_epochs), desc="Epoch"):
  #     tr_loss = 0
  #     nb_tr_examples, nb_tr_steps = 0, 0
  #     for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
  #       batch = tuple(t.to(device) for t in batch)
  #       input_ids, input_mask, segment_ids, label_ids = batch

  #       # define a new function to compute loss values for both output_modes
  #       logits = model(input_ids, segment_ids, input_mask, labels=None)

  #       if output_mode == "classification":
  #         loss_fct = CrossEntropyLoss()
  #         loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
  #       elif output_mode == "regression":
  #         loss_fct = MSELoss()
  #         loss = loss_fct(logits.view(-1), label_ids.view(-1))

  #       if n_gpu > 1:
  #         loss = loss.mean() # mean() to average on multi-gpu.
  #       if args.gradient_accumulation_steps > 1:
  #         loss = loss / args.gradient_accumulation_steps

  #       if args.fp16:
  #         optimizer.backward(loss)
  #       else:
  #         loss.backward()

  #       tr_loss += loss.item()
  #       nb_tr_examples += input_ids.size(0)
  #       nb_tr_steps += 1
  #       if (step + 1) % args.gradient_accumulation_steps == 0:
  #         if args.fp16:
  #           # modify learning rate with special warm up BERT uses
  #           # if args.fp16 is False, BertAdam is used that handles this automatically
  #           lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
  #           for param_group in optimizer.param_groups:
  #             param_group['lr'] = lr_this_step
  #         optimizer.step()
  #         optimizer.zero_grad()
  #         global_step += 1

  # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
  #   # Save a trained model, configuration and tokenizer
  #   model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

  #   # If we save using the predefined names, we can load using `from_pretrained`
  #   output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
  #   output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

  #   torch.save(model_to_save.state_dict(), output_model_file)
  #   model_to_save.config.to_json_file(output_config_file)
  #   tokenizer.save_vocabulary(args.output_dir)

  #   # Load a trained model and vocabulary that you have fine-tuned
  #   model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
  #   tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
  # else:
  #   model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
  # model.to(device)

  # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
  #   eval_examples = processor.get_dev_examples(args.data_dir)
  #   eval_features = convert_examples_to_features(
  #     eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
  #   logger.info("***** Running evaluation *****")
  #   logger.info("  Num examples = %d", len(eval_examples))
  #   logger.info("  Batch size = %d", args.eval_batch_size)
  #   all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  #   all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  #   all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

  #   if output_mode == "classification":
  #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
  #   elif output_mode == "regression":
  #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

  #   eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  #   # Run prediction for full data
  #   eval_sampler = SequentialSampler(eval_data)
  #   eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

  #   model.eval()
  #   eval_loss = 0
  #   nb_eval_steps = 0
  #   preds = []

  #   for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
  #     input_ids = input_ids.to(device)
  #     input_mask = input_mask.to(device)
  #     segment_ids = segment_ids.to(device)
  #     label_ids = label_ids.to(device)

  #     with torch.no_grad():
  #       logits = model(input_ids, segment_ids, input_mask, labels=None)

  #     # create eval loss and other metric required by the task
  #     if output_mode == "classification":
  #       loss_fct = CrossEntropyLoss()
  #       tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
  #     elif output_mode == "regression":
  #       loss_fct = MSELoss()
  #       tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
      
  #     eval_loss += tmp_eval_loss.mean().item()
  #     nb_eval_steps += 1
  #     if len(preds) == 0:
  #       preds.append(logits.detach().cpu().numpy())
  #     else:
  #       preds[0] = np.append(
  #         preds[0], logits.detach().cpu().numpy(), axis=0)

  #   eval_loss = eval_loss / nb_eval_steps
  #   preds = preds[0]
  #   if output_mode == "classification":
  #     preds = np.argmax(preds, axis=1)
  #   elif output_mode == "regression":
  #     preds = np.squeeze(preds)
  #   result = compute_metrics(task_name, preds, all_label_ids.numpy())
  #   loss = tr_loss/global_step if args.do_train else None

  #   result['eval_loss'] = eval_loss
  #   result['global_step'] = global_step
  #   result['loss'] = loss

  #   output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
  #   with open(output_eval_file, "w") as writer:
  #     logger.info("***** Eval results *****")
  #     for key in sorted(result.keys()):
  #       logger.info("  %s = %s", key, str(result[key]))
  #       writer.write("%s = %s\n" % (key, str(result[key])))

  #   # hack for MNLI-MM
  #   if task_name == "mnli":
  #     task_name = "mnli-mm"
  #     processor = processors[task_name]()

  #     if os.path.exists(args.output_dir + '-MM') and os.listdir(args.output_dir + '-MM') and args.do_train:
  #       raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
  #     if not os.path.exists(args.output_dir + '-MM'):
  #       os.makedirs(args.output_dir + '-MM')

  #     eval_examples = processor.get_dev_examples(args.data_dir)
  #     eval_features = convert_examples_to_features(
  #       eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
  #     logger.info("***** Running evaluation *****")
  #     logger.info("  Num examples = %d", len(eval_examples))
  #     logger.info("  Batch size = %d", args.eval_batch_size)
  #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
  #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

  #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  #     # Run prediction for full data
  #     eval_sampler = SequentialSampler(eval_data)
  #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

  #     model.eval()
  #     eval_loss = 0
  #     nb_eval_steps = 0
  #     preds = []

  #     for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
  #       input_ids = input_ids.to(device)
  #       input_mask = input_mask.to(device)
  #       segment_ids = segment_ids.to(device)
  #       label_ids = label_ids.to(device)

  #       with torch.no_grad():
  #         logits = model(input_ids, segment_ids, input_mask, labels=None)
      
  #       loss_fct = CrossEntropyLoss()
  #       tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
      
  #       eval_loss += tmp_eval_loss.mean().item()
  #       nb_eval_steps += 1
  #       if len(preds) == 0:
  #         preds.append(logits.detach().cpu().numpy())
  #       else:
  #         preds[0] = np.append(
  #           preds[0], logits.detach().cpu().numpy(), axis=0)

  #     eval_loss = eval_loss / nb_eval_steps
  #     preds = preds[0]
  #     preds = np.argmax(preds, axis=1)
  #     result = compute_metrics(task_name, preds, all_label_ids.numpy())
  #     loss = tr_loss/global_step if args.do_train else None

  #     result['eval_loss'] = eval_loss
  #     result['global_step'] = global_step
  #     result['loss'] = loss

  #     output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
  #     with open(output_eval_file, "w") as writer:
  #       logger.info("***** Eval results *****")
  #       for key in sorted(result.keys()):
  #         logger.info("  %s = %s", key, str(result[key]))
  #         writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
  main()
