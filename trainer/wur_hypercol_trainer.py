import torch
import torch.nn.functional as F
from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor

from torch.nn.utils import clip_grad_value_
import torch.autograd as autograd
from torch.autograd import Variable

from apex import amp
from apex.fp16_utils import *

import numpy as np

DEFAULT_LOSS_WEIGHTS = {
    "match": 1,
    "spatial_softmax": 0,
    "heatmap_l1": 0,
}

class WURHypercolTrainer:
    def __init__(self, models, optimizers, loss_fns, logger, config):
        self.opts = config.trainer_config
        self.search_domain = self.opts['search_domain'] if 'search_domain' in self.opts else "A"
        self.loss_weights = self.opts['loss_weights'] if 'loss_weights' in self.opts else DEFAULT_LOSS_WEIGHTS.copy()
        self.hm_act = F.tanhshrink if 'hm_act' in self.opts else False
        self.fp16 = False
        self.binarize = False

        self.log_eval_batch = False
        
        self.FtsA = self._model_precision(models["FtsA"])
        self.FtsB = self._model_precision(models["FtsB"])
            
        if optimizers:
            if "Fts" in optimizers:
                self.optim_fts = self._optim_precision(optimizers["Fts"])
            else:
                self.optim_fts = None
            
        if loss_fns:
            self.Lmatch = loss_fns["Lmatch"][0]

        self.device = torch.device(config.device)
        self.logger = logger
        self.log_freq = config.log_freq
        self.attached = {}
        self.curr_epoch = 0
        self.log_str = ""

    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _optim_precision(self, optim):
        if self.fp16:
            return FP16_Optimizer(optim, dynamic_loss_scale=True)
        else:
            return optim

    def _model_precision(self, model):
        if self.fp16:
            return network_to_half(model)
        else:
            return model

    def _prepare_batch(self, batch, non_blocking=True):
        xs, ys = batch
        
        if isinstance(xs, list):
            xs = [convert_tensor(x, self.device, non_blocking=non_blocking).float() for x in xs]
        else:
            xs = [convert_tensor(xs, self.device, non_blocking=non_blocking).float()]

        if isinstance(ys, list):
            ys = [convert_tensor(y, self.device, non_blocking=non_blocking).float() for y in ys]
        else:
            ys = [convert_tensor(ys, self.device, non_blocking=non_blocking).float()]

        if self.fp16:
            xs = [x.half() for x in xs]
            ys = [y.half() for y in ys]

        return xs, ys

    def _zero_grad(self):
        if self.optim_fts:
            self.optim_fts.zero_grad()
        
    def freeze_model(self, model):
        model.eval()
        for params in model.parameters():
            params.requires_grad = False
        
    def unfreeze_model(self, model):
        model.train()
        for params in model.parameters():
            params.requires_grad = True

    def _AdamW(self, optimizer, wd=1e-2):
        if self.adamW and isinstance(optimizer, torch.optim.Adam):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-wd * group['lr'], param.data)

    def l2_shift_loss(self, heatmap, label, device="cuda"):
        hm = heatmap.view(heatmap.size(0), -1)
        gt = label.view(label.size(0), -1)

        hm_max, hm_pos = hm.max(1)
        gt_max, gt_pos = gt.max(1)

        hm_pos = torch.Tensor( np.unravel_index(hm_pos.cpu().numpy(), heatmap.shape[2:]) ).transpose(0, 1)
        gt_pos = torch.Tensor( np.unravel_index(gt_pos.cpu().numpy(), label.shape[2:]) ).transpose(0, 1)

        l2 = F.pairwise_distance(hm_pos, gt_pos)
        return l2.to(device)

    # Apply softmax per channel
    def spatial_softmax(self, heatmap):
        b,c,h,w = heatmap.size()
        x = heatmap.view(b, c, -1).transpose(2, 1)
        x = F.softmax(x, dim=1)
        return x.transpose(2, 1).view(b, c, h, w)

    # Similar to Neighbourhood Consensus Network Loss
    # For the distribution at each layer to be Kronecker Delta
    def softmax_localization_loss(self, heatmap, alpha=0):
        heatmap = self.spatial_softmax(heatmap)
        b,c,h,w = heatmap.size()

        # Flatten the tensor but keep the channels for adaptions to multiscale later
        scores, _ = torch.max(heatmap.view(b, c, -1), dim=2)
        return torch.mean(scores) + alpha*heatmap.sum()

    def weighted_binary_cross_entropy(self, heatmap, labels, thresh=0.8, device="cuda", reduction="mean"):
        b, c, h, w = heatmap.shape
        weight = torch.sum(labels < thresh)/(torch.sum(labels >= thresh))

        return self.Lmatch(heatmap, labels, pos_weight=weight.to(device), reduction=reduction)

    def train(self, engine, batch):
        self.FtsA.train()
        self.FtsB.train()

        log_str = ""
        curr_step = self.logger.counters["train"]

        # Extract the a and b image pairs from the batch and whether the pairs match
        (search_img, template_img, template_hard), y = self._prepare_batch(batch)
        
        self._zero_grad()

        y_a = self.FtsA(search_img)
        y_bhn = self.FtsB(template_hard)
                
        heatmap_hneg_raw = self.FtsA.correlation_map(y_a, y_bhn, self.hm_act)

        if self.loss_weights["spatial_softmax"] > 0:
            heatmap_hneg = self.spatial_softmax(heatmap_hneg_raw)
        else:
            heatmap_hneg = heatmap_hneg_raw

        if engine.state.iteration % 1000 == 0:
            self.logger.log_image_grid("Search", search_img, "train")
            self.logger.log_image_grid("Template_match", template_hard, "train")
            self.logger.log_image_grid("Heatmap", heatmap_hneg, "train")
            self.logger.log_image_grid("Heatmap_raw", heatmap_hneg_raw, "train")
            self.logger.log_image_grid("Ground_Truth", y[0], "train")

        match_loss = self.weighted_binary_cross_entropy(heatmap_hneg, y[0], device=self.device)

        self.logger.add_scalars('train/match_loss', {'Lmatch': match_loss.item()}, curr_step)
        
        heatmap_l1_loss = self.loss_weights["heatmap_l1"]*heatmap_hneg_raw.norm(p=1)

        self.logger.add_scalars('train/regularize', {'L1': heatmap_l1_loss}, curr_step)

        loss = self.loss_weights["match"]*match_loss + heatmap_l1_loss

        self.logger.add_scalar('train/loss', loss.item(), curr_step)

        log_str += "Ltotal: {:.5f} \t".format(loss.item())
        log_str += "Lmatch: {:.5f} \t".format(match_loss.item())

        loss.backward()
        self.optim_fts.step()

        self.log_str = log_str

        return None, None

    def on_epoch_start(self, engine, phase=None):
        if phase == "train":
            self.curr_epoch = engine.state.epoch
        
        if phase == "evaluate":
            self.log_eval_batch = True         

    def on_epoch_end(self, engine, phase=None):
        if phase in ["evaluate", "test"]:
            metrics = engine.state.metrics
            log = ""
            for k, v in metrics.items():
                log += "{}: {:.5f}  ".format(k, v)

            print("{} Results - Epoch: {}  {}".format(phase.capitalize(), self.curr_epoch, log))

    def on_iteration_start(self, engine, phase=None):            
        if phase == "train":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["train_loader"]) + 1

            if curr_iter % self.log_freq == 0:
                print("Epoch[{}] Iteration[{}/{}] {}".format(engine.state.epoch, curr_iter, len(self.attached["train_loader"]), self.log_str))
        
        elif phase == "test":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["test_loader"]) + 1
            if curr_iter % self.log_freq == 0:
                print("Iteration[{}/{}]".format(curr_iter, len(self.attached["test_loader"])))

    def on_iteration_end(self, engine, phase=None):
        pass

    def infer_batch(self, batch):
        self.FtsA.eval()
        self.FtsB.eval()

        with torch.no_grad():
            imgs, y = self._prepare_batch(batch)

            try:
                (search_img, template_img, template_hard) = imgs
            except:
                (search_img, template_img, template_hard, _) = imgs

            y_a = self.FtsA(search_img)
            y_b = self.FtsB(template_img)
            y_bhn = self.FtsB(template_hard)
            y_bn = torch.roll(y_bhn, -1, 0)

            # Create negative examples for matching and easy negative heatmaps
            heatmap_neg_raw = self.FtsA.correlation_map(y_a, y_bn, self.hm_act)
            heatmap_hneg_raw = self.FtsA.correlation_map(y_a, y_bhn, self.hm_act)

            if self.loss_weights["spatial_softmax"] > 0:
                heatmap_hneg = self.spatial_softmax(heatmap_hneg_raw)
                heatmap_neg = self.spatial_softmax(heatmap_neg_raw)
            else:
                heatmap_hneg = heatmap_hneg_raw
                heatmap_neg = heatmap_neg_raw

            d_a, attn_a = None, None
            d_b, attn_b = None, None
            d_bhn, attn_bhn = None, None

            hms = (heatmap_neg, heatmap_neg_raw, heatmap_hneg, heatmap_hneg_raw)
            fts = (y_a, y_b, y_bhn)
            dets = (d_a, attn_a, d_b, attn_b, d_bhn, attn_bhn)

            return imgs, hms, y, fts, dets
        
    def evaluate(self, engine, batch):
        curr_step = self.logger.counters["evaluate"]

        imgs, hms, y, fts, dets = self.infer_batch(batch)
        (search_img, template_img, template_hard) = imgs
        (heatmap_neg, heatmap_neg_raw, heatmap_hneg, heatmap_hneg_raw) = hms
        (y_a, y_b, y_bhn) = fts
        (d_a, attn_a, d_b, attn_b, d_bhn, attn_bhn) = dets

        match_loss = self.weighted_binary_cross_entropy(heatmap_hneg, y[0], device=self.device)

        self.logger.add_scalars('evaluate/match_loss', {'Lmatch': match_loss.item()}, curr_step)
     
        heatmap_l1_loss = self.loss_weights["heatmap_l1"]*heatmap_hneg_raw.norm(p=1)

        self.logger.add_scalars('evaluate/regularize', {'L1': heatmap_l1_loss}, curr_step)

        loss = self.loss_weights["match"]*match_loss + heatmap_l1_loss
        self.logger.add_scalar('evaluate/loss', loss.item(), curr_step)

        if self.log_eval_batch:
            self.logger.log_image_grid("ev_Search", search_img, "evaluate")
            self.logger.log_image_grid("ev_Template", template_img, "evaluate")
            self.logger.log_image_grid("ev_Template_s", template_hard, "evaluate")
            self.logger.log_image_grid("ev_Heatmap", heatmap_hneg, "evaluate")
            self.logger.log_image_grid("ev_Ground_Truth", y[0], "evaluate")
            self.log_eval_batch = False

        return None, None #(search_img, template_img), (heatmap_pos, heatmap_neg)

    def attach(self, name, obj):
        self.attached[name] = obj
