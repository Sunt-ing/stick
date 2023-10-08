from .op import *
from .tensor_op import *

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

def make_tuple(*args):
    return MakeTensorTuple()(*args)


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
                from .. import init
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)

def tuple_get_item(value, index):
    return TupleGetItem(index)(value)

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
