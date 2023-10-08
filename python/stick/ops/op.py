

from typing import Any, List, Optional, Tuple, Union
import numpy, time
from ..dtr import Dtr, ENABLE_DTR
from ..backend_selection import *


TENSOR_COUNTER = 0
LAZY_MODE = False
ENABLE_CHECKPOINT = False
ENABLE_GRAD = True

# introduced to impl checkpointing
class Function:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()
    
    def gradient(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class Op(Function):
    """Operator definition."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray], **kwargs):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    # def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
    def gradient(self, out_grad, node):
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    # def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
    def gradient_as_tuple(self, out_grad, node):
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class Value:
    """A value in the computational graph."""

    op: Optional[Function]
    inputs: List["Value"]
    # The following fields are cached fields for dynamic computation.
    # Can be used to impl checkpointing / rematerialization.
    outputs: NDArray
    requires_grad: bool

    # return the data size in GPU
    def internal_size(self, data):
        if isinstance(data, NDArray):
            return data.internal_size()
        
        # assert isinstance(data, tuple)
        # assert isinstance(self.op, MakeTensorTuple) or isinstance(self.op, Split)

        size = 0
        for datum in data:
            size += datum.internal_size()
        return size

    def get_outputs(self):
        """Run compute to realize the cached data"""
        if self.outputs is not None:
            # add trace
            if ENABLE_DTR:
                return Dtr.get_obj(self)
            else:
                return self.outputs
        # naive checkpointing
        if ENABLE_CHECKPOINT and not ENABLE_GRAD:
            return self.op.compute(*[x.get_outputs() for x in self.inputs])
        # DTR
        elif ENABLE_DTR:
            inputs = [x.get_outputs() for x in self.inputs]

            start = time.perf_counter()
            self.outputs = self.op.compute(*inputs)
            end = time.perf_counter()

            mem1 = self.internal_size(self.outputs)
            cost = end - start

            Dtr.add(self, end, mem1, cost)
        # normal
        else:
            self.outputs = self.op.compute(*[x.get_outputs() for x in self.inputs])
            return self.outputs

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Function],
        # inputs: List["Tensor"],
        inputs: List["Value"],
        *,
        num_outputs: int = 1,
        outputs: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.outputs = outputs
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            outputs=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Function, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.get_outputs()
        return value

    def numpy(self):
        data = self.get_outputs()
        if array_api is numpy:
            return data
        return data.numpy() if not isinstance(data, tuple) else [x.numpy() for x in data]
