# tests for hw4

import sys
sys.path.append('./python')
import numpy as np
import pytest

import stick as stk
from stick import NDArray


np.random.seed(2)


_DEVICES = [stk.cpu(), pytest.param(stk.cuda(),
    marks=pytest.mark.skipif(not stk.cuda().enabled(), reason="No GPU"))]


TRAIN = [True, False]
@pytest.mark.parametrize("train", TRAIN)
def test_cifar10_dataset(train):
    dataset = stk.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))
    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 32, 32)


BATCH_SIZES = [1, 15]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_cifar10_loader(batch_size, train, device):
    cifar10_train_dataset = stk.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = stk.data.DataLoader(cifar10_train_dataset, batch_size)
    for (X, y) in train_loader:
        break
    assert isinstance(X.realize_cached_data(), NDArray)
    assert isinstance(X, stk.Tensor)
    assert isinstance(y, stk.Tensor)
    assert X.dtype == 'float32'


BPTT = [3, 32]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("bptt", BPTT)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ptb_dataset(batch_size, bptt, train, device):
    corpus = stk.data.Corpus("data/ptb")
    print(corpus)
    if train:
        data = stk.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    else:
        data = stk.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
    X, y = stk.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
    assert X.shape == (bptt, batch_size)
    assert y.shape == (bptt * batch_size,)
    assert isinstance(X, stk.Tensor)
    assert X.dtype == 'float32'
    assert X.device == device
    assert isinstance(X.realize_cached_data(), NDArray)
    ntokens = len(corpus.dictionary)
    assert ntokens == 10000
