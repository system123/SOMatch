from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class CategoricalAccuracyOneHot(Metric):
    """
    Calculates the categorical accuracy.
    `update` must receive output of the form (y_pred, y).
    `y_pred` must be in the following shape (batch_size, num_categories, ...)
    `y` must be in the following shape (batch_size, num_categories, ...)
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        _, indices = torch.max(y_pred, dim=1)
        _, labels = torch.max(y, dim=1)
        correct = torch.eq(indices, labels).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
