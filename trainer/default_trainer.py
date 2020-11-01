import torch
import torch.nn.functional as F
from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor

from utils.helpers import BinaryClassificationMeter, accuracy

class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, logger, config):
        self.opts = config.trainer_config

        self.model = model["Model"]

        if optimizer:
            self.optimizer = optimizer["Optim"]
        
        if loss_fn:
            self.loss_fn = loss_fn["Loss"][0]

        self.logger = logger
        self.device = config.device
        self.log_freq = config.log_freq
        self.attached = {}
        self.curr_epoch = 0
        self.metric = BinaryClassificationMeter()
        self.metric_train = BinaryClassificationMeter()

    def _prepare_batch(self, batch):
        xs, ys = batch

        if isinstance(xs, list):
            xs = [convert_tensor(x, self.device).float() for x in xs]
        else:
            xs = [convert_tensor(xs, self.device).float()]

        if isinstance(ys, list):
            ys = [convert_tensor(y, self.device).float() for y in ys]
        else:
            ys = [convert_tensor(ys, self.device).float()]

        return xs, ys

    def train(self, engine, batch):
        self.model.train()

        curr_step = self.logger.counters["train"]

        self.optimizer.zero_grad()

        xs, ys = self._prepare_batch(batch)
        y_pred = self.model(*xs)

        if not (isinstance(y_pred, list) or isinstance(y_pred, tuple)):
            ys = ys[0]

        loss = self.loss_fn(y_pred, ys, pos_weight=torch.Tensor([1.5]).to("cuda"))

        self.logger.add_scalars('train/loss', {'L': loss.item()}, curr_step)

        if engine.state.iteration % 1000 == 0:
            self.logger.log_image_grid("Input", xs[0], "train")
            y_img = torch.ones_like(xs[0])*ys.view(ys.size(0),1,1,1)
            self.logger.log_image_grid("Label", y_img, "train", normalize=False)
            y2_img = torch.ones_like(xs[0])*torch.sigmoid(y_pred).view(y_pred.size(0),1,1,1)
            self.logger.log_image_grid("Prediction", y2_img, "train", normalize=False)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def on_epoch_start(self, engine, phase=None):
        self.log_batch = True
        self.metric_train.reset()
        if phase == "train":
            self.curr_epoch = engine.state.epoch

    def on_epoch_end(self, engine, phase=None):
        if phase in ["evaluate", "test"]:
            metrics = engine.state.metrics
            log = ""
            for k, v in metrics.items():
                log += "{}: {:.2f}  ".format(k, v)

            print("{} Results - Epoch: {}  {}".format(phase.capitalize(), self.curr_epoch, log))
        
        if phase in ["evaluate"]:
            curr_step = self.logger.counters["evaluate"]
            self.logger.add_scalars('evaluate/metrics', {'Acc': self.metric.acc, 'Precision': self.metric.pre, 'f1':self.metric.f1, 'Recall': self.metric.rec}, curr_step)
            self.metric.reset()

    def on_iteration_start(self, engine, phase=None):
        if phase == "train":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["train_loader"]) + 1
            if curr_iter % self.log_freq == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, curr_iter, len(self.attached["train_loader"]), engine.state.output))
        elif phase == "test":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["test_loader"]) + 1
            if curr_iter % self.log_freq == 0:
                print("Iteration[{}/{}]".format(curr_iter, len(self.attached["test_loader"])))

    def on_iteration_end(self, engine, phase=None):
        pass

    def infer_batch(self, batch):
        self.model.eval()

        with torch.no_grad():
            xs, ys = self._prepare_batch(batch)
            y_pred = self.model(*xs)

            return xs, ys, y_pred
        
    def evaluate(self, engine, batch):
        curr_step = self.logger.counters["evaluate"]

        xs, ys, y_pred = self.infer_batch(batch)

        if not (isinstance(y_pred, list) or isinstance(y_pred, tuple)):
            ys = ys[0]

        if self.log_batch:
            self.logger.log_image_grid("evInput", xs[0], "evaluate")
            y_img = torch.ones_like(xs[0])*ys.view(ys.size(0),1,1,1)
            self.logger.log_image_grid("evLabel", y_img, "evaluate", normalize=False)
            y2_img = torch.ones_like(xs[0])*torch.sigmoid(y_pred).view(y_pred.size(0),1,1,1)
            self.logger.log_image_grid("evPrediction", y2_img, "evaluate", normalize=False)
            self.log_batch = False

        loss = self.loss_fn(y_pred, ys)

        self.metric.update(torch.sigmoid(y_pred), ys)

        self.logger.add_scalars('evaluate/loss', {'L': loss.item()}, curr_step)

        return y_pred.float(), ys.float()

    def attach(self, name, obj):
        self.attached[name] = obj
