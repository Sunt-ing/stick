"""The NN modules."""

from . import init, ops
from .ops import Tensor
import numpy as np
from typing import List

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


# trace all the fields of the Parameter type
def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
            out += ops.broadcast_to(self.bias, out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        len = 1
        for shape in X.shape[1:]:
            len *= shape
        return X.reshape((X.shape[0], len))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        return (ops.logsumexp(logits, (1,)) / logits.shape[0]).sum() - (logits * init.one_hot(logits.shape[1], y, device=logits.device) / logits.shape[0]).sum()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device))
        self.running_mean = init.zeros(dim, device=device) 
        self.running_var = init.ones(dim, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            Ex = (x.sum(axes=(0,)) / x.shape[0])
            Ex_vec = Ex.reshape((1, x.shape[1]))
            Varx = (((x + (-Ex_vec).broadcast_to(x.shape)) ** 2).sum(axes=(0,)) / x.shape[0])
            Varx_vec= Varx.reshape((1, x.shape[1]))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * Ex.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Varx.data
            norm = (x + (-Ex_vec).broadcast_to(x.shape)) / ((Varx_vec.broadcast_to(x.shape) + self.eps) ** 0.5)
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        else:
            norm = (x + (-self.running_mean).broadcast_to(x.shape)) / ((self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5)
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Ex = (x.sum(axes=(1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        Varx = (((x + (-Ex)) ** 2).sum(axes=(1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        norm = (x + (-Ex)) / ((Varx + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        randrop = init.randb(x.shape[0], x.shape[1], p=1 - self.p, device=x.device, dtype=x.dtype)
        return x * randrop / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (self.kernel_size - 1) // 2
        self.weight = Parameter(init.kaiming_uniform(
            in_channels * kernel_size * kernel_size, out_channels * kernel_size * kernel_size, 
            shape=(kernel_size, kernel_size, in_channels, out_channels), 
            device=device, 
            requires_grad=True))
        if bias:
            bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound, device=device, requires_grad=True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        _x = x.transpose((1, 2))
        _x = _x.transpose((2, 3)) # NCHW -> NHWC
        conv_res = ops.conv(_x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            conv_res += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(conv_res.shape)
        return conv_res.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1.0 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == 'tanh':
            self.nonlinearity = Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = ReLU()
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        if self.bias:
            return self.nonlinearity(X @ self.W_ih + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)) + 
                                     h @ self.W_hh + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)))
        else:
            return self.nonlinearity(X @ self.W_ih + h @ self.W_hh)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)] + \
                         [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype) for _ in range(num_layers - 1)]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_split = list(tuple(ops.split(X, axis=0)))
        if h0 is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=self.device, dtype=self.dtype)
        h_n = list(tuple(ops.split(h0, axis=0)))
        for num_layer in range(self.num_layers):
            h_t = h_n[num_layer]
            for i in range(X.shape[0]):
                X_split[i] = self.rnn_cells[num_layer](X_split[i], h_t)
                h_t = X_split[i]
            h_n[num_layer] = h_t
        return ops.stack(X_split, axis=0), ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1.0 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0 = h[0]
            c0 = h[1]
        if self.bias:
            gate_input = (X @ self.W_ih + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size)) +
                          h0 @ self.W_hh + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size)))
        else:
            gate_input = (X @ self.W_ih + h0 @ self.W_hh)
        gate_input_split = tuple(ops.split(gate_input, axis=1))
        i, f, g, o = Sigmoid()(ops.stack(gate_input_split[0 : self.hidden_size], axis=1)), \
                    Sigmoid()(ops.stack(gate_input_split[self.hidden_size : 2 * self.hidden_size], axis=1)), \
                    Tanh()(ops.stack(gate_input_split[2 * self.hidden_size:3 * self.hidden_size], axis=1)), \
                    Sigmoid()(ops.stack(gate_input_split[3 * self.hidden_size:4 * self.hidden_size], axis=1))
        c_ = f * c0 + i * g
        h_ = o * Tanh()(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)] + \
                         [LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype) for _ in range(num_layers - 1)]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_split = list(tuple(ops.split(X, axis=0)))
        if h is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0 = h[0]
            c0 = h[1]
        h_n = list(tuple(ops.split(h0, axis=0)))
        c_n = list(tuple(ops.split(c0, axis=0)))
        for num_layer in range(self.num_layers):
            h_t = h_n[num_layer]
            c_t = c_n[num_layer]
            for i in range(X.shape[0]):
                X_split[i], c_t = self.lstm_cells[num_layer](X_split[i], (h_t, c_t))
                h_t = X_split[i]
            h_n[num_layer] = h_t
            c_n[num_layer] = c_t
        return ops.stack(X_split, axis=0), (ops.stack(h_n, axis=0), ops.stack(c_n, axis=0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype, requires_grad=True))
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_one_hot = init.one_hot(self.weight.shape[0], x, device=self.device, dtype=self.dtype).reshape((seq_len * bs, self.weight.shape[0]))
        return (x_one_hot @ self.weight).reshape((seq_len, bs, self.weight.shape[1]))
        ### END YOUR SOLUTION
