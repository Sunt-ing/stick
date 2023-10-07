import sys

sys.path.append("./python")
sys.path.append("./tests")
import stick as stk
from stick import Dtr, ops
from test_nn_and_optim import train_epoch_1
import numpy as np
import pytest

_DEVICES = [stk.cpu(), pytest.param(stk.cuda(),
    marks=pytest.mark.skipif(not stk.cuda().enabled(), reason="No GPU"))]

@pytest.mark.parametrize("device", _DEVICES, ids=["cuda", "cpu"])
def test_dtr(device):
    ops.ENABLE_DTR = True
    # max is 174780036
    Dtr.mem_limit = 100000000
    np.random.seed(1)
    np.testing.assert_allclose(train_epoch_1(5, 250, stk.optim.Adam, lr=0.01, weight_decay=0.1),
        np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)
    

if __name__ == "__main__":
    test_dtr(device="cpu")
