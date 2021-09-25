# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import copy

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage

from detectron2.engine import TrainerBase

class BankTrainer(TrainerBase):
	"""
	BankTrainer for efficient N-way K-shot training.
	It mostly follows SimpeTrainer but has 2 major differences.

	1. Update EMA model per every iteration
	2. Use EMA model to get detached latent vector from image

	A simple trainer for the most common type of task:
	single-cost single-optimizer single-data-source iterative optimization,
	optionally using data-parallelism.
	It assumes that every step, you:

	1. Compute the loss with a data from the data_loader.
	2. Compute the gradients with the above loss.
	3. Update the model with the optimizer.

	All other tasks during training (checkpointing, logging, evaluation, LR schedule)
	are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

	If you want to do anything fancier than this,
	either subclass TrainerBase and implement your own `run_step`,
	or write your own training loop.
	"""

	def __init__(self, model, ema_model, memory, data_loader, optimizer, ema_m=0.999):
		"""
		Args:
			model: a torch Module. Takes a data from data_loader and returns a
				dict of losses.
			data_loader: an iterable. Contains data to be used to call model.
			optimizer: a torch optimizer.
		"""
		super().__init__()

		"""
		We set the model to training mode in the trainer.
		However it's valid to train a model that's in eval mode.
		If you want your model (or a submodule of it) to behave
		like evaluation during training, you can overwrite its train() method.
		"""
		model.train()

		self.model = model
		self.ema_model = ema_model
		self.ema_m = ema_m
		self.data_loader = data_loader
		self._data_loader_iter = iter(data_loader)
		self.optimizer = optimizer
		self.memory = memory
		self.sample_rate = 5

	@torch.no_grad()
	def ema_model_update(self):
		for param_train, param_eval in zip():
			param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))
		
		for buffer_train, buffer_eval in zip(self.model.buffers(), self.ema_model.buffers()):
			buffer_eval.copy_(buffer_train)

	def run_step(self):
		"""
		Implement the standard training logic described above.
		"""
		assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
		start = time.perf_counter()
		"""
		If you want to do something with the data, you can wrap the dataloader.
		"""

		data = next(self._data_loader_iter)
		data_time = time.perf_counter() - start

		gt_class = [x['instances'].gt_classes for x in data]
		gt_class = torch.cat(gt_class).unique(sorted=True)
		whole_class = list(self.memory.mapping.keys())
		neg_class = torch.tensor([x for x in whole_class if x not in gt_class])
		gt_mask = torch.randperm(len(gt_class))[:self.sample_rate]
		neg_mask = torch.randperm(len(neg_class))[:self.sample_rate]
		gt_class = torch.cat([gt_class[gt_mask], neg_class[neg_mask]])
		memory_feature_dict = self.memory(gt_class=gt_class)

		"""
		If you want to do something with the losses, you can wrap the model.
		"""
		#_, loss_wo_memory_dict = self.model(data)
		_, loss_dict= self.model(data, memory_feature_dict)

		"""
		for k, v in loss_wo_memory_dict.items():
			loss_dict["wo_mem_{}".format(k)] = v
		"""

		if isinstance(loss_dict, torch.Tensor):
			losses = loss_dict
			loss_dict = {"total_loss": loss_dict}
		else:
			losses = sum(loss_dict.values())

		"""
		If you need to accumulate gradients or do something similar, you can
		wrap the optimizer with your custom `zero_grad()` method.
		"""
		self.optimizer.zero_grad()
		losses.backward()

		self._write_metrics(loss_dict, data_time)

		"""
		If you need gradient clipping/scaling or other processing, you can
		wrap the optimizer with your custom `step()` method. But it is
		suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
		"""
		self.optimizer.step()
		self.ema_model_update()

		"""
		detach feature dict in here and gather feature dict along all batch
		"""
		# TODO_P: detach feature_dict in here
		with torch.no_grad():
			self.model.prepare_feature = True
			curr_batch_feature_dict, _ = self.model(data)
			self.memory(feature_dict=curr_batch_feature_dict)
			self.model.prepare_feature = False

	def _write_metrics(
		self,
		loss_dict: Mapping[str, torch.Tensor],
		data_time: float,
		prefix: str = "",
	) -> None:
		BankTrainer.write_metrics(loss_dict, data_time, prefix)

	@staticmethod
	def write_metrics(
		loss_dict: Mapping[str, torch.Tensor],
		data_time: float,
		prefix: str = "",
	) -> None:
		"""
		Args:
			loss_dict (dict): dict of scalar losses
			data_time (float): time taken by the dataloader iteration
			prefix (str): prefix for logging keys
		"""
		metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
		metrics_dict["data_time"] = data_time

		# Gather metrics among all workers for logging
		# This assumes we do DDP-style training, which is currently the only
		# supported method in detectron2.
		all_metrics_dict = comm.gather(metrics_dict)

		if comm.is_main_process():
			storage = get_event_storage()

			# data_time among workers can have high variance. The actual latency
			# caused by data_time is the maximum among workers.
			data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
			storage.put_scalar("data_time", data_time)

			# average the rest metrics
			metrics_dict = {
				k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
			}
			total_losses_reduced = sum(metrics_dict.values())
			if not np.isfinite(total_losses_reduced):
				raise FloatingPointError(
					f"Loss became infinite or NaN at iteration={storage.iter}!\n"
					f"loss_dict = {metrics_dict}"
				)

			storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
			if len(metrics_dict) > 1:
				storage.put_scalars(**metrics_dict)

	def state_dict(self):
		ret = super().state_dict()
		ret["optimizer"] = self.optimizer.state_dict()
		return ret

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.optimizer.load_state_dict(state_dict["optimizer"])
