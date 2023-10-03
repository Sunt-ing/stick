# tests for hw4

import sys
sys.path.append('./python')
import numpy as np
import pytest
from stick import backend_ndarray as nd
import stick as stk


_DEVICES = [stk.cpu(), pytest.param(stk.cuda(),
    marks=pytest.mark.skipif(not stk.cuda().enabled(), reason="No GPU"))]

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(stk.Tensor(c, device=args[0].device), out)
    if isinstance(backward_grad[0], stk.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


stack_back_params = [
    ( (3, 4), 3, 0),
    ( (3, 4), 3, 1),
    ( (3, 4), 3, 2),
    ( (3, 4), 5, 2),
    ( (3, 4), 1, 2),
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_stack_backward(shape, n, axis, device):
    np.random.seed(0)
    get_tensor = lambda shape: stk.Tensor(np.random.randn(*shape)*5, device=device)
    backward_check(stk.stack, [get_tensor(shape) for _ in range(n)], axis=axis)


stack_params = [
    {"shape": (10,3),    "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", stack_params)
def test_stack_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_stack_stk = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_stk += [stk.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = stk.stack(to_stack_stk, axis=axis)


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
]
@pytest.mark.parametrize("device", [nd.cpu()])
@pytest.mark.parametrize("params", pad_params)
def test_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params['shape'], params['padding']
    _A = np.random.randn(*shape)
    _B = np.pad(_A, padding)
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)

    assert np.linalg.norm(B.numpy() - _B) < 1e-4


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_forward_params)
def test_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    _A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = stk.Tensor(_A, device=device)
    B = stk.flip(A, axes=axes)

    assert np.linalg.norm(B.numpy() - _B) < 1e-4


flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (2, 3, 3, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 4), "axes": (0,1)},
    {"shape": (2, 3, 3, 4), "axes": (1,2)},
    {"shape": (3, 3, 6, 4), "axes": (1,2)},
    {"shape": (2, 3, 3, 4), "axes": (2,3)},
    {"shape": (3, 3, 6, 4), "axes": (2,3)},
    {"shape": (2, 3, 3, 4), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_backward_params)
def test_flip_backward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    backward_check(stk.flip, stk.Tensor(np.random.randn(*shape), device=device), axes=axes)


# @pytest.mark.parametrize("device", _DEVICES)
# def test_init_calculate_fans(device):
#     _A = np.random.randn(3, 3, 16, 8)
#     A = stk.Tensor(_A, device=device)
#     assert stk.init._calculate_fans(A) == (144, 72)

#     _A = np.random.randn(3, 3, 16, 8)
#     A = stk.Tensor(_A, device=device)
#     assert stk.init._calculate_fans(A) == (144, 72)


#     _A = np.random.randn(16, 8)
#     A = stk.Tensor(_A, device=device)
#     assert stk.init._calculate_fans(A) == (16, 8)


@pytest.mark.parametrize("device", _DEVICES)
def test_init_kaiming_uniform(device):
    _A = np.random.randn(3, 3, 16, 8)
    A = stk.Tensor(_A, device=device)
    np.random.seed(0)
    A = stk.init.kaiming_uniform(16*9, 8*9, shape=A.shape)
    assert abs(A.sum().numpy() - -2.5719218) < 1e-4


@pytest.mark.parametrize("device", _DEVICES)
def test_resnet9(device):
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device)

    assert num_params(model) == 431946

    _A = np.random.randn(2, 3, 32, 32)
    A = stk.Tensor(_A, device=device)
    y = model(A)

    assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
         2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
       [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
         1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2



@pytest.mark.parametrize("device", _DEVICES)
def test_dilate_forward(device):
    np.random.seed(0)
    device = stk.cpu()

    _A = np.random.randint(1, 10, size=(2, 5))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=0, axes=(0,)).numpy() - np.array([[6., 1., 4., 4., 8.],
       [4., 6., 3., 5., 8.]])) < 1e-5 

    _A = np.random.randint(1, 10, size=(2, 5))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=1, axes=(0,)).numpy() - np.array([[7., 9., 9., 2., 7.],
       [0., 0., 0., 0., 0.],
       [8., 8., 9., 2., 6.],
       [0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=1, axes=(1,)).numpy() - np.array([[9., 0., 5., 0., 4., 0., 1., 0., 4., 0.],
       [6., 0., 1., 0., 3., 0., 4., 0., 9., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=1, axes=(0,1)).numpy() - np.array([[2., 0., 4., 0., 4., 0., 4., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 2., 0., 1., 0., 5., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=2, axes=(0,1)).numpy() - np.array([[4., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [8., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
    A = stk.Tensor(_A, device=device)
    assert np.linalg.norm(stk.dilate(A, dilation=1, axes=(1,2)).numpy() - np.array([[[[1., 1.],
         [0., 0.],
         [5., 6.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[6., 7.],
         [0., 0.],
         [9., 5.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]],


       [[[2., 5.],
         [0., 0.],
         [9., 2.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[2., 8.],
         [0., 0.],
         [4., 7.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]]])) < 1e-5


dilate_backward_params = [
    {"shape": (2, 5),          "d": 1, "axes": (0,)},
    {"shape": (2, 5),          "d": 2, "axes": (1,)},
    {"shape": (2, 5),          "d": 1, "axes": (0,1)},
    {"shape": (2, 5),          "d": 0, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 2, "axes": (0,1)},
    {"shape": (3, 3, 6, 4),     "d": 3, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 0, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (1,2)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (2,3)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (2,3)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", dilate_backward_params)
def test_dilate_backward(params, device):
    np.random.seed(0)
    shape, d, axes = params['shape'], params['d'], params['axes']
    backward_check(stk.dilate, stk.Tensor(np.random.randn(*shape), device=device), dilation=d, axes=axes)


def test_stack_vs_pytorch():
    np.random.seed(0)
    import torch
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(5, 5)
    D = np.random.randn(15, 5)

    Andl = stk.Tensor(A, requires_grad=True)
    Bndl = stk.Tensor(B, requires_grad=True)
    Cndl = stk.Tensor(C, requires_grad=True)
    Dndl = stk.Tensor(D, requires_grad=True)

    Atch = torch.tensor(A, requires_grad=True)
    Btch = torch.tensor(B, requires_grad=True)
    Ctch = torch.tensor(C, requires_grad=True)
    Dtch = torch.tensor(D, requires_grad=True)

    Xndl = stk.stack([Andl, Cndl @ Bndl, Cndl], axis=1)
    Xtch = torch.stack([Atch, Ctch @ Btch, Ctch], dim=1)

    assert Xndl.shape == Xtch.shape
    assert np.linalg.norm(Xndl.numpy() - Xtch.detach().numpy()) < 1e-3

    Yndl = (Dndl @ Xndl.reshape((5, 15)) @ Dndl).sum()
    Ytch = (Dtch @ Xtch.reshape(5, 15) @ Dtch).sum()

    assert np.linalg.norm(Yndl.numpy() - Ytch.detach().numpy()) < 1e-3

    Yndl.backward()
    Ytch.backward()

    assert np.linalg.norm(Andl.grad.cached_data.numpy() - Atch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Bndl.grad.cached_data.numpy() - Btch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Cndl.grad.cached_data.numpy() - Ctch.grad.detach().numpy()) < 1e-3



conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = stk.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = stk.init.rand(10, cin, s, s, device=device)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())

    assert np.linalg.norm(f(x).cached_data.numpy() - g(z).data.numpy()) < 1e-3


conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = stk.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = stk.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"


op_conv_shapes = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]
@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = stk.Tensor(_Z, device=device)
    W = stk.Tensor(_W, device=device)
    y = stk.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)


@pytest.mark.parametrize("device", _DEVICES)
def test_train_cifar10(device):
    np.random.seed(0)
    dataset = stk.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = stk.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=False
             # collate_fn=stk.data.collate_ndarray,
             # drop_last=False,
             # device=device,
             # dtype="float32"
             )
    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, opt=stk.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
    assert np.linalg.norm(np.array([out[0], out[1][0]]) - np.array([0.09375, 3.5892258])) < 1e-2


def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=stk.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X,y = stk.Tensor(X, device=device), stk.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)


######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:64], A.shape)


def Rand(*shape, device=stk.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=10, size=shape)
    return stk.Tensor(_A, device=device)


def RandC(*shape, entropy=1):
    if stk.cuda().enabled():
        return Rand(*shape, device=stk.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")
