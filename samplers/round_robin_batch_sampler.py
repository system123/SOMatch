from torch.utils.data import Sampler, BatchSampler
from torch._six import int_classes as _int_classes
from itertools import islice, cycle

# Samples batches in an alternating manner when there are different sections to a dataset, drop_last is True by default
# If samplers are of different length then the round robin will only occur while there are sufficient samples remaining, after which it will return from the longer sampler
# list(RoundRobinBatchSampler([SubsetRandomSampler(range(10,20)), SubsetRandomSampler(range(100,120))], batch_size=3))
# [[15, 10, 11], [119, 103, 104], [14, 17, 16], [107, 109, 102], [18, 12, 19], [112, 105, 114], [117, 108, 100], [111, 116, 101], [118, 113, 110]]
class RoundRobinBatchSampler(Sampler):
    def __init__(self, samplers, batch_size):
        for sampler in samplers:
            if not isinstance(sampler, Sampler):
                raise ValueError("sampler should be an instance of torch.utils.data.Sampler, but got sampler={}".format(sampler))
            if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
                raise ValueError("batch_size should be a positive integeral value, but got batch_size={}".format(batch_size))

        self.samplers = samplers
        self.batch_size = batch_size
        self.batch_samplers = [BatchSampler(sampler, self.batch_size, True) for sampler in self.samplers]

    def __iter__(self):
        num_active = len(self.batch_samplers)
        nexts = cycle(iter(it).__next__ for it in self.batch_samplers)

        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    def __len__(self):
        return sum([len(sampler) // self.batch_size for sampler in self.samplers])