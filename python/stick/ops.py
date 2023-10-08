"""Basic operators and automatic differentiation."""
from typing import Any, List, Optional, Tuple, Union
import numpy, time
from .backend_selection import *
from .dtr import Dtr

LAZY_MODE = False
TENSOR_COUNTER = 0
ENABLE_GRAD = True
CHECKPOINT_MODE = False
ENABLE_DTR = False


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

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
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

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
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
        
        assert isinstance(data, tuple)
        assert isinstance(self.op, MakeTensorTuple) or isinstance(self.op, Split)

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
        
        # note: data implicitly calls realized cached data
        inputs = [x.get_outputs() for x in self.inputs]
        start = time.perf_counter()
        data = self.op.compute(*inputs)
        cost = time.perf_counter() - start

        # ENABLE_GRAD is controlled by checkpointing
        if (not CHECKPOINT_MODE) or (self.requires_grad and ENABLE_GRAD):
            self.outputs = data
        
        if ENABLE_DTR:
            ts = time.perf_counter()
            mem = self.internal_size(data)
            Dtr.add(self, ts, mem, cost)
        return data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Function],
        inputs: List["Tensor"],
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


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype="float32",
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                outputs = array.get_outputs()
            else:
                # fall back, copy through numpy conversion
                outputs = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            outputs = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            outputs=outputs,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if (not tensor.requires_grad) or (CHECKPOINT_MODE and not ENABLE_GRAD):
                return tensor.detach()
            tensor.get_outputs()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            outputs=data
                if not isinstance(data, Tensor)
                else data.get_outputs(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        # not DTR as the time cost is unknown and the data can be unrecoverable
        self.outputs = value.get_outputs()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.get_outputs())

    @property
    def shape(self):
        return self.get_outputs().shape

    @property
    def dtype(self):
        return self.get_outputs().dtype

    @property
    def device(self):
        data = self.get_outputs()
        if array_api is numpy:
            return default_device()
        return data.device

    # auto differentiate
    def backward(self, out_grad=None):
        from . import init
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "stick.Tensor(" + str(self.get_outputs()) + ")"

    def __str__(self):
        return self.get_outputs().__str__()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        ### BEGIN YOUR SOLUTION
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return PowerScalar(other)(self)
        ### END YOUR SOLUTION

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)
       
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(Negate()(self), other)
        else:
            return AddScalar(other)(-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.get_outputs()
        return len(cdata)

    def __getitem__(self, index: int):
        return tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "stick.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.get_outputs())


class TensorOp(Op):
    """ Op class specialized to output tensors, will be alterate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    

class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                from . import init
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)



def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited:
        return
    for pred in node.inputs:
        topo_sort_dfs(pred, visited, topo_order)
    visited.add(node)
    topo_order.append(node)
    ### END YOUR SOLUTION

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)

def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Value):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    node_to_output_grads_list: dict[Tensor, List[Tensor]] = {}
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        # grad sum is a must, grad back-prop is optional since this can be a leaf node
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if node.op is None:
            continue
        # compute grad for each input
        gradients = node.op.gradient_as_tuple(node.grad, node)
        for input, grad in zip(node.inputs, gradients):
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = []
            node_to_output_grads_list[input].append(grad)
    ### END YOUR SOLUTION

class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -lhs * out_grad / (rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        order = list(range(len(a.shape)))
        if self.axes is None:
            order[-1] = order[-2]
            order[-2] = len(order) - 1
        else:
            order[self.axes[0]] = self.axes[1]
            order[self.axes[1]] = self.axes[0]
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        len_diff = len(out_grad.shape) - len(node.inputs[0].shape)
        agg_axis = list(range(0, len_diff))
        for i in range(len_diff, len(out_grad.shape)):
            if node.inputs[0].shape[i - len_diff] != out_grad.shape[i]:
                agg_axis.append(i)
        return summation(out_grad, tuple(agg_axis)).reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # self.axes can be int
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(self.axes)
        for i, axis in enumerate(sorted(list(self.axes))):
            # -i because each sum() operation will reduce the dimension number
            a = a.sum(axis - i)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return broadcast_to(out_grad, node.inputs[0].shape)
        output_shape = list(node.inputs[0].shape)
        if type(self.axes) is int:
            output_shape[self.axes] = 1
        else:
            for i in self.axes:
                output_shape[i] = 1
        return broadcast_to(out_grad.reshape(tuple(output_shape)), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lgrad.shape) > len(lhs.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rgrad.shape) > len(rhs.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.get_outputs() > 0, dtype="float32", device=node.device, requires_grad=False)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(self.axes, keepdims=True)
        ret = array_api.log(array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)) + maxz
        if self.axes is None:
            axes = list(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = list(self.axes)
        
        if self.axes is not None:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in axes]
        else:
            out_shape = [1]
        
        return ret.reshape(tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_max_dim = Tensor(z.get_outputs().max(self.axes, keepdims=True), device=z.device, requires_grad=False)
        z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
        z_exp_sum = summation(z_exp, axes=self.axes)
        grad_z_exp_sum = out_grad / z_exp_sum
        ori_shape = z.shape
        sum_shape = range(len(z.shape)) if self.axes is None else self.axes
        now_shape = list(ori_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        arr_len = len(args)
        curr_shape = list(args[0].shape)
        curr_shape.insert(self.axis, arr_len)
        stack_arr = array_api.empty(curr_shape, device=args[0].device)
        slice_idx = [slice(0, len) for len in curr_shape]
        for i in range(arr_len):
            slice_idx[self.axis] = i
            stack_arr[tuple(slice_idx)] = args[i]
        return stack_arr
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        arr_len = A.shape[self.axis]
        curr_shape = list(A.shape)
        split_arr = []
        slice_idx = [slice(0, len) for len in curr_shape]
        curr_shape.pop(self.axis)
        for i in range(arr_len):
            slice_idx[self.axis] = i
            split_arr.append(array_api.reshape(A[tuple(slice_idx)].compact(), curr_shape))
        return tuple(split_arr)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 1. construct and full a matrix of the new size
        # 2. copy the previous matrix to the corresponding places in the new one
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        new_array = array_api.full(tuple(new_shape), 0, device=a.device)
        ranges = [slice(0, shape) for shape in new_shape]
        for axis in self.axes:
            ranges[axis] = slice(0, new_shape[axis], self.dilation + 1)
        new_array[tuple(ranges)] = a
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ranges = [slice(0, shape) for shape in a.shape]
        for axis in self.axes:
            ranges[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(ranges)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


# Reference: 
#   https://github.com/Somoku/CMU10-714-DLSys
#   https://github.com/YuanchengFang/dlsys_solution


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        strided_A = A.as_strided(shape=(N, (H - K + 1) // self.stride, (W - K + 1) // self.stride, K, K, C_in),
                                 strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact(). \
                                reshape((N * (H - K + 1) // self.stride * (W - K + 1) // self.stride, inner_dim))
        out = strided_A @ B.compact().reshape((K * K * C_in, C_out))
        return out.compact().reshape((N, (H - K + 1) // self.stride, (W - K + 1) // self.stride, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        # out_grad: N * (H + 2P - K + 1) // self.stride * (W + 2P - K + 1) // self.stride * C_out
        # W: K * K * C_in * C_out
        # W_transpose: K * K * C_out * C_in
        # X_grad: N * H * W * C_in

        # X: N * H * W * C_in
        # out_grad: N * (H + 2P - K + 1) * (W + 2P - K + 1) * C_out
        # W_grad: K * K * C_in * C_out

        X = node.inputs[0]
        W = node.inputs[1]
        K, _, _, _ = W.shape
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1) # N * (H + 2P - K + 1) * (W + 2P - K + 1) * C_out
        W_flip = flip(W, (0, 1)) # K * K * C_in * C_out
        W_transpose = transpose(W_flip, (2, 3)) # K * K * C_out * C_in
        X_grad = conv(out_grad, W_transpose, padding=K - 1 - self.padding)

        X_permute = transpose(X, (0, 3)) # C_in * H * W * N
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H + 2P - K + 1) * (W + 2P - K + 1) * N * C_out
        W_grad_transpose = conv(X_permute, out_grad_permute, padding=self.padding) # C_in * K * K * C_out
        W_grad = transpose(transpose(W_grad_transpose, (0, 1)), (1, 2)) # K * K * C_in * C_out
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
