import os

import torch
import numpy as np
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from datetime import datetime
from utils.helpers import get_learning_rate

class TensorboardLogger:

	def __init__(self, log_every=10, log_params=False, log_dir=None, log_images=False, log_grads=False, **kwargs):
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.log_dir = os.path.join(log_dir, "runs", current_time)
		self.writer = SummaryWriter(log_dir=self.log_dir)

		self.counters = {"evaluate": 0, "train": 0, "test": 0}
		self.epochs = {"evaluate": 0, "train": 0, "test": 0}
		self.log_every = log_every
		self.log_params = log_params if isinstance(log_params, bool) else False
		self.log_images = log_images if isinstance(log_images, bool) else False
		self.log_grads = log_grads if isinstance(log_grads, bool) else False

		print(f"Logger: Log parameters={log_params}, Log gradients={log_grads}")

	# def state_dict(self):
	# 	state = {}
	# 	state['counters'] = self.counters
	# 	state['epochs'] = self.epochs 
	# 	return {'state': state}

	def fast_forward(self, last_epoch=0, step_per_epoch=0):
		step = (last_epoch+1)*step_per_epoch
		self.counters = {"evaluate": step, "train": step, "test": step}
		self.epochs = {"evaluate": last_epoch+1, "train": last_epoch+1, "test": last_epoch+1}

	def teardown(self):
		self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
		self.writer.close()

	def add_embedding(self, features, images, phase="train", stage="epoch"):
		step = self.epochs[phase] if stage == "epoch" else self.counters[phase]
		self.writer.add_embedding(features, label_img=images, global_step=step)

	def _plot_metrics(self, metrics, phase, step):
		for m_name, m_val in metrics.items():
			self.writer.add_scalar("{}/{}".format(phase, m_name), m_val, step)

	def log_gradients(self, tag, model, phase="train", log_every=1000):
		if (self.log_grads is True) and (self.counters[phase] % log_every == 0):
			for name, param in model.named_parameters():
				if param.grad is not None:
					self.writer.add_histogram("{}_{}".format(tag, name), param.grad.data.cpu().numpy(), self.counters[phase])

	def log_preactivations(self, module, phase="train"):
		classname = module.__class__.__name__

		def _log_preactivations(input, output):
			self.writer.add_histogram("{}_{}".format(classname, "forward"), output.data.cpu().numpy(), self.counters[phase])

		if classname.find('Conv') != -1 or classname.find('Linear') != -1:
			module.register_forward_hook(_log_preactivations)

	def log_image_grid(self, name, images, phase="train", normalize=True):
		if self.log_images is True:
			x_rg = vutils.make_grid(images, normalize=normalize, scale_each=True)
			self.writer.add_image(name, x_rg, self.counters[phase])

	# Method Missing - automatically assume it is for the summaryWriter
	def __getattr__(self, method_name):
		log_fn = getattr(self.writer, method_name, None)

		if log_fn:
			return log_fn
		else:
			raise AttributeError(method_name)

	def log_iteration(self, engine, phase="train", models=None, optims=None):
		# other_metrics = {}
		if optims:
			for name, optim in optims.items():
				lr = get_learning_rate(optim)[0]
				self.writer.add_scalar("{}/{}_lr".format(phase, name), lr, self.counters[phase])

		if self.counters[phase] % self.log_every == 0:
			self._plot_metrics(engine.state.metrics, phase, self.counters[phase])
			# self._plot_metrics(other_metrics, phase, self.counters[phase])

		self.counters[phase] += 1

	def log_epoch(self, engine, phase="train", models=None, optims=None):
		self._plot_metrics(engine.state.metrics, phase, self.counters[phase])

		if phase == "train" and self.log_params is True:
			for m_name, model in models.items():
				for name, param in model.named_parameters():
					self.writer.add_histogram("{}_{}".format(m_name, name), param.data.cpu().numpy(), self.epochs[phase])

		if phase == "evaluate":
			self.epochs[phase] += 1
		else:
			self.epochs[phase] = engine.state.epoch