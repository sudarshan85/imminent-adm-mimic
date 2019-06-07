#!/usr/bin/env python

import logging
import csv
import datetime
import numpy as np

from argparse import Namespace
from collections import OrderedDict

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, Timer
from ignite.contrib.handlers import ProgressBar

from .containers import ModelContainer, DataContainer

class IgniteTrainer(object):
  def __init__(self, model_container: ModelContainer, data_container: DataContainer, args:
      Namespace, metrics: OrderedDict, early_stop: bool=False, log_training: bool=True, verbose=False) -> None:

    self.logger = logging.getLogger(__name__)
    logging_level = logging.DEBUG if verbose else logging.WARN
    self.logger.setLevel(logging_level)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    self.logger.addHandler(sh)

    self.mc = model_container
    self.dc = data_container
    self.metrics = metrics
    self.log_training = log_training
    self.early_stop = early_stop

    # retrieve arguments from args
    self.device = args.device
    self.n_epochs = args.n_epochs
    self.modeldir = args.workdir/'models'
    self.training_log = args.workdir/'training_log.csv'
    self.cp_prefix = args.checkpointer_prefix
    self.cp_name = args.checkpointer_name
    self.cp_save_every = args.checkpointer_save_every
    self.cp_save_total = args.checkpointer_save_total
    self.best_val_loss = np.inf
    self.best_model_name = None

    if self.early_stop:
      if 'early_stop_patience' not in args:
        raise AttributeError(f"Early Stopping enabled but 'early_stop_patience' attribute not in args")
      else:
        self.es_patience = args.early_stop_patience

    # create trainers and evaluators
    self.trainer = create_supervised_trainer(self.mc.model, self.mc.optimizer, self.mc.loss_fn, device=self.device)
    self.train_eval = create_supervised_evaluator(self.mc.model, metrics=self.metrics, device=self.device)
    self.val_eval = create_supervised_evaluator(self.mc.model, metrics=self.metrics, device=self.device)

    # set loss to be shown in progress bar
    RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'loss')
    self.pbar = ProgressBar(persist=True)
    self.pbar.attach(self.trainer, ['loss'])

    # setup timers
    self.epoch_timer = Timer(average=True)
    self.epoch_timer.attach(self.trainer, start=Events.EPOCH_STARTED, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED, step=Events.ITERATION_COMPLETED)
    self.training_timer = Timer()
    self.training_timer.attach(self.trainer)

    self._add_handlers()

  def _add_handlers(self):
    # csv logger
    if self.log_training:
      self.logger.debug("Creating CSV logger handler")
      self.trainer.add_event_handler(Events.STARTED, self._open_csv)
      self.trainer.add_event_handler(Events.COMPLETED, self._close_csv)

    # epoch logger
    self.logger.debug("Creating Epoch Logger")
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)

    # checkpointer
    self.logger.debug("Creating Checkpointer")
    checkpointer = ModelCheckpoint(self.modeldir, self.cp_prefix, require_empty=False, save_interval=self.cp_save_every, n_saved=self.cp_save_total, save_as_state_dict=True)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.cp_name: self.mc.model})

    # early stopping
    if self.early_stop:
      self.logger.debug("Creating Early Stopper")
      early_stopper = EarlyStopping(self.es_patience, self.score_fn, self.trainer)
      self.val_eval.add_event_handler(Events.COMPLETED, early_stopper)

    # ReduceLR if plateau step
    if self.mc.reduce_lr is not None:
      self.logger.debug("Creating ReduceLROnPlateau Scheduler")
      self.val_eval.add_event_handler(Events.EPOCH_COMPLETED, self._reduce_lr_step)

  def _open_csv(self, engine):
    self.fp = open(self.training_log, 'w')
    self.writer = csv.writer(self.fp)
    row = ['epoch']
    for metric in self.metrics.keys():
      row.append('training_' + metric)
    for metric in self.metrics.keys():
      row.append('validation_' + metric)
    self.writer.writerow(row)

  def _reduce_lr_step(self, engine):
    self.mc.reduce_lr.step(engine.state.metrics['loss'])

  def _close_csv(self, engine):
    train_time = str(datetime.timedelta(seconds=self.training_timer.value()))
    self.logger.info(f"Training done. Total training time: {train_time}")
    self.fp.write(f"{train_time}\n")
    self.fp.close()

  def _log_epoch(self, engine):
    self.epoch_timer.reset()

    # compute metrics on training and validation datasets
    self.train_eval.run(self.dc.train_dl)
    self.val_eval.run(self.dc.val_dl)
    epoch = engine.state.epoch
    row = [epoch]

    # print out metrics
    s = 'Training '
    for metric in self.metrics.keys():
      s += f'{metric} {self.train_eval.state.metrics[metric]:0.3f} '
      row.append(f'{self.train_eval.state.metrics[metric]:0.3f}')
    s += '\n'
    s += 'Validation '
    for metric in self.metrics.keys():
      s += f'{metric} {self.val_eval.state.metrics[metric]:0.3f} '
      row.append(f'{self.val_eval.state.metrics[metric]:0.3f}')
    self.pbar.log_message(s)

    # write metrics to csv file
    if self.log_training:
      self.writer.writerow(row)

    if self.best_val_loss > self.val_eval.state.metrics['loss']:
      self.best_val_loss = self.val_eval.state.metrics['loss']
      self.best_model_name = self.cp_prefix + '_' + self.cp_name + '_' + str(epoch) + '.pth'

  @staticmethod
  def score_fn(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss

  def run(self):
    self.logger.info(f"Running {self.mc.model.__class__.__name__} for {self.n_epochs} epochs on device {self.device} with batch size {self.dc.bs}")
    self.trainer.run(self.dc.train_dl, self.n_epochs)
    return self.best_model_name
