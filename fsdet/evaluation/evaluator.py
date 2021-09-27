import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.structures.boxes import Boxes
from detectron2.utils.events import EventStorage, TensorboardXWriter, get_event_storage
import torch
from detectron2.utils.visualizer import Visualizer
import numpy as np
from detectron2.utils.comm import is_main_process
from PIL import Image
import copy
import cv2


class DatasetEvaluator:
	"""
	Base class for a dataset evaluator.

	The function :func:`inference_on_dataset` runs the model over
	all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

	This class will accumulate information of the inputs/outputs (by :meth:`process`),
	and produce evaluation results in the end (by :meth:`evaluate`).
	"""

	def reset(self):
		"""
		Preparation for a new round of evaluation.
		Should be called before starting a round of evaluation.
		"""
		pass

	def process(self, input, output):
		"""
		Process an input/output pair.

		Args:
			input: the input that's used to call the model.
			output: the return value of `model(output)`
		"""
		pass

	def evaluate(self):
		"""
		Evaluate/summarize the performance, after processing all input/output pairs.

		Returns:
			dict:
				A new evaluator class can return a dict of arbitrary format
				as long as the user can process the results.
				In our train_net.py, we expect the following format:

				* key: the name of the task (e.g., bbox)
				* value: a dict of {metric name: score}, e.g.: {"AP50": 80}
		"""
		pass


class DatasetEvaluators(DatasetEvaluator):
	def __init__(self, evaluators):
		assert len(evaluators)
		super().__init__()
		self._evaluators = evaluators

	def reset(self):
		for evaluator in self._evaluators:
			evaluator.reset()

	def process(self, input, output):
		for evaluator in self._evaluators:
			evaluator.process(input, output)

	def evaluate(self):
		results = OrderedDict()
		for evaluator in self._evaluators:
			result = evaluator.evaluate()
			if is_main_process():
				for k, v in result.items():
					assert (
						k not in results
					), "Different evaluators produce results with the same key {}".format(k)
					results[k] = v
		return results


def condition_on_dataset(model, data_loader, memory):
	num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
	logger = logging.getLogger(__name__)
	logger.info("Start Condition on {} images".format(len(data_loader.dataset)))

	total = len(data_loader)  # inference data loader must have a fixed length

	logging_interval = 50
	num_warmup = min(5, logging_interval - 1, total - 1)
	start_time = time.time()
	total_compute_time = 0
	model.prepare_feature = True

	with torch.no_grad():
		for idx, inputs in enumerate(data_loader):
			if idx == num_warmup:
				start_time = time.time()
				total_compute_time = 0

			start_compute_time = time.time()
			feature_dict, _= model(inputs)
			torch.cuda.synchronize()
			total_compute_time += time.time() - start_compute_time
			memory(feature_dict)

			if (idx + 1) % logging_interval == 0:
				duration = time.time() - start_time
				seconds_per_img = duration / (idx + 1 - num_warmup)
				eta = datetime.timedelta(
					seconds=int(seconds_per_img * (total - num_warmup) - duration)
				)
				logger.info(
					"Condition done {}/{}. {:.4f} s / img. ETA={}".format(
						idx + 1, total, seconds_per_img, str(eta)
					)
				)

	model.prepare_feature = False

def log_prediction(inputs, outputs, meta, idx):
	storage = get_event_storage()
	pred_iter = storage.iter
	base = len(inputs)
	for i, (x, pred) in enumerate(zip(inputs, outputs)):
		img = x["image"].permute(1, 2, 0)
		if meta.input_format == "BGR":
			img = img[:, :, [2, 1, 0]]
		else:
			img = np.asarray(
				Image.fromarray(img, mode=meta.input_format).convert(
					"RGB"
				)
			)

		pred_img_size = pred['instances']._image_size[::-1]
		org_img_size = tuple(img.shape[:2])[::-1]

		pred_img = copy.deepcopy(img)
		pred_img = torch.tensor(cv2.resize(pred_img.numpy(), dsize=pred_img_size, interpolation=cv2.INTER_CUBIC))

		gt_img = Visualizer(img, metadata=meta)
		pred_img = Visualizer(pred_img, metadata=meta)

		target_fields = x["instances"].get_fields()
		labels = [
			meta.thing_classes[i]
			for i in target_fields["gt_classes"]
		]
		gt_vis = gt_img.overlay_instances(
			labels=labels,
			boxes=target_fields.get("gt_boxes", None),
		)

		target_fields = pred["instances"].get_fields()
		pred_labels = [
			meta.thing_classes[i]
			for i in target_fields["pred_classes"]
		]
		pred_boxes = Boxes(target_fields.get("pred_boxes", None).tensor.cpu())
		pred_vis = pred_img.overlay_instances(
			labels=pred_labels[:10],
			boxes=pred_boxes[:10],
		)

		gt_image = gt_vis.get_image()
		pred_image = pred_vis.get_image()
		pred_image = cv2.resize(pred_image, dsize=org_img_size, interpolation=cv2.INTER_CUBIC)
		eval_image = np.concatenate((gt_image, pred_image), axis=0)

		storage.iter = idx * base + i
		storage.put_image('{}/eval_vis'.format(meta.name), eval_image.transpose(2, 0, 1))

	storage.iter = pred_iter


def meta_inference_on_dataset(model, feature_dict, data_loader, evaluator, prop_evaluator, meta, writer):
	"""
	Run model on the data_loader and evaluate the metrics with evaluator.
	The model will be used in eval mode.

	Args:
		model (nn.Module): a module which accepts an object from
			`data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

			If you wish to evaluate a model in `training` mode instead, you can
			wrap the given model and override its behavior of `.eval()` and `.train()`.
		data_loader: an iterable object with a length.
			The elements it generates will be the inputs to the model.
		evaluator (DatasetEvaluator): the evaluator to run. Use
			:class:`DatasetEvaluators([])` if you only want to benchmark, but
			don't want to do any evaluation.

	Returns:
		The return value of `evaluator.evaluate()`
	"""
	num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
	logger = logging.getLogger(__name__)
	logger.info("Start inference on {} images".format(len(data_loader)))

	total = len(data_loader)  # inference data loader must have a fixed length
	evaluator.reset()
	prop_evaluator.reset()

	logging_interval = 50
	num_warmup = min(5, logging_interval - 1, total - 1)
	start_time = time.time()
	total_compute_time = 0
	with inference_context(model), torch.no_grad():
		for idx, inputs in enumerate(data_loader):
			if idx == num_warmup:
				start_time = time.time()
				total_compute_time = 0

			start_compute_time = time.time()
			outputs, prop = model(inputs, feature_dict)
			if idx % 50 == 0:
				log_prediction(inputs, outputs, meta, idx)
				writer.write()
			torch.cuda.synchronize()
			total_compute_time += time.time() - start_compute_time
			evaluator.process(inputs, outputs)
			prop_evaluator.process(inputs, prop)

			if (idx + 1) % logging_interval == 0:
				duration = time.time() - start_time
				seconds_per_img = duration / (idx + 1 - num_warmup)
				eta = datetime.timedelta(
					seconds=int(seconds_per_img * (total - num_warmup) - duration)
				)
				logger.info(
					"Inference done {}/{}. {:.4f} s / img. ETA={}".format(
						idx + 1, total, seconds_per_img, str(eta)
					)
				)
				#break

	# Measure the time only for this worker (before the synchronization barrier)
	total_time = int(time.time() - start_time)
	total_time_str = str(datetime.timedelta(seconds=total_time))
	# NOTE this format is parsed by grep
	logger.info(
		"Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
			total_time_str, total_time / (total - num_warmup), num_devices
		)
	)
	total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
	logger.info(
		"Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
			total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
		)
	)

	results = evaluator.evaluate()
	prop_results = prop_evaluator.evaluate()
	# An evaluator may return None when not in main process.
	# Replace it by an empty dict instead to make it easier for downstream code to handle
	if results is None:
		results = {}
	return results, prop_results


def inference_on_dataset(model, data_loader, evaluator):
	"""
	Run model on the data_loader and evaluate the metrics with evaluator.
	The model will be used in eval mode.

	Args:
		model (nn.Module): a module which accepts an object from
			`data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

			If you wish to evaluate a model in `training` mode instead, you can
			wrap the given model and override its behavior of `.eval()` and `.train()`.
		data_loader: an iterable object with a length.
			The elements it generates will be the inputs to the model.
		evaluator (DatasetEvaluator): the evaluator to run. Use
			:class:`DatasetEvaluators([])` if you only want to benchmark, but
			don't want to do any evaluation.

	Returns:
		The return value of `evaluator.evaluate()`
	"""
	num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
	logger = logging.getLogger(__name__)
	logger.info("Start inference on {} images".format(len(data_loader)))

	total = len(data_loader)  # inference data loader must have a fixed length
	evaluator.reset()

	logging_interval = 50
	num_warmup = min(5, logging_interval - 1, total - 1)
	start_time = time.time()
	total_compute_time = 0
	with inference_context(model), torch.no_grad():
		for idx, inputs in enumerate(data_loader):
			if idx == num_warmup:
				start_time = time.time()
				total_compute_time = 0

			start_compute_time = time.time()
			outputs = model(inputs)
			torch.cuda.synchronize()
			total_compute_time += time.time() - start_compute_time
			evaluator.process(inputs, outputs)

			if (idx + 1) % logging_interval == 0:
				duration = time.time() - start_time
				seconds_per_img = duration / (idx + 1 - num_warmup)
				eta = datetime.timedelta(
					seconds=int(seconds_per_img * (total - num_warmup) - duration)
				)
				logger.info(
					"Inference done {}/{}. {:.4f} s / img. ETA={}".format(
						idx + 1, total, seconds_per_img, str(eta)
					)
				)

	# Measure the time only for this worker (before the synchronization barrier)
	total_time = int(time.time() - start_time)
	total_time_str = str(datetime.timedelta(seconds=total_time))
	# NOTE this format is parsed by grep
	logger.info(
		"Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
			total_time_str, total_time / (total - num_warmup), num_devices
		)
	)
	total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
	logger.info(
		"Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
			total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
		)
	)

	results = evaluator.evaluate()
	# An evaluator may return None when not in main process.
	# Replace it by an empty dict instead to make it easier for downstream code to handle
	if results is None:
		results = {}
	return results


@contextmanager
def inference_context(model):
	"""
	A context where the model is temporarily changed to eval mode,
	and restored to previous mode afterwards.

	Args:
		model: a torch Module
	"""
	training_mode = model.training
	model.eval()
	yield
	model.train(training_mode)
