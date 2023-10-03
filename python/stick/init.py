import math
from .backend_selection import default_device
from .ops import Tensor


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = default_device() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = default_device() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = default_device() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = default_device() if device is None else device
    return Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    if not shape:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    if not shape:
        shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3.0 / fan_in)
    if not shape:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    if not shape:
        shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION
