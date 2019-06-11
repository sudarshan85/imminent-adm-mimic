#!/usr/bin/env python

import torch
from typing import List, Tuple

from pytorch_pretrained_bert import BertAdam

def build_optimizer(named_params: List[Tuple[int, torch.nn.parameter.Parameter]], n_steps: int, lr: float, warmup_prop: float, wd: float, schedule='warmup_linear') -> BertAdam: 
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  grouped_params = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  
  return BertAdam(grouped_params, lr=lr, warmup=warmup_prop, t_total=n_steps, schedule=schedule, weight_decay=wd)
