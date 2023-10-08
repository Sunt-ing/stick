from .basic import *


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

