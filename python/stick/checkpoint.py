# Reference: 
#   idea: https://arxiv.org/abs/1604.06174
#   impl: https://github.com/Lyken17/pytorch-memonger

from . import ops
from .ops import Tensor, Function
from .nn import Module, BatchNorm1d
import numpy as np
from typing import Any
import math


class enable_grad:
    def __init__(self, enable) -> None:
        self.prev = False
        self.enable = enable

    def __enter__(self) -> None:
        self.prev = ops.ENABLE_GRAD
        ops.ENABLE_GRAD = self.enable

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ops.ENABLE_GRAD = self.prev


class Checkpoint(Function):
    def __call__(self, *inputs):
        return self.compute(*inputs)

    def __init__(self, modules):
        self.modules = modules
        self.input = None
        # currently, all randomness comes from numpy
        self.fwd_cpu_rng_state = None
        # store the state of BatchNorm op to make it idempotent
        self.running_mean_dict = {}
        self.running_var_dict = {}

    # forward
    def compute(self, input):
        self.input = input
        self.fwd_cpu_rng_state = np.random.get_state()
        for i, m in enumerate(self.modules):
            # store the state of BatchNorm
            if isinstance(m, BatchNorm1d):
                self.running_mean_dict[i] = m.running_mean
                self.running_var_dict[i] = m.running_var

        # disable grad to discard the intermediate results
        with enable_grad(False):
            outputs = self.run_module(input)
        return outputs

    # backward
    def gradient(self, out_grad, node):
        np.random.set_state(self.fwd_cpu_rng_state)
        for i, m in enumerate(self.modules):
            if isinstance(m, BatchNorm1d):
                m.running_mean = self.running_mean_dict[i]
                m.running_var = self.running_var_dict[i]
        with enable_grad(True):
            input = self.input.detach()
            input.requires_grad = self.input.requires_grad
            self.input = None # reduce memory footprint
            output = self.run_module(input) #recompute
            ops.compute_gradient_of_variables(output, out_grad)
        return input.grad
    
    def run_module(self, input):
        for module in self.modules:
            input = module(input)
        return input


def checkpoint(modules, input):
    return Checkpoint(modules)(input)


class Memonger(Module):
    def __init__(self, *modules):
        ops.CHECKPOINT_MODE = True
        super().__init__()
        self.modules = modules

    def forward(self, input: Tensor) -> Tensor:
        # To make the granularity finer, we can overide __setattr__ to track 
        # the execution order of children modules, as in PyTorch.
        # See: https://discuss.pytorch.org/t/how-does-module-children-know-how-to-sort-submodules/58473/2
        seg_size = int(math.sqrt(len(self.modules)))
        seg_num = len(self.modules) // seg_size
        end = 0
        for start in range(0, seg_size * (seg_num - 1), seg_size):
            end = start + seg_size
            input = checkpoint(self.modules[start: end], input)
        # last segment will be specifically handled even if the modules len is aligned
        return checkpoint(self.modules[end: len(self.modules)], input)
