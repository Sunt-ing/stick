import sys, time
sys.path.append("./python")
sys.path.append("./tests")
import stick as stk
from stick import dtr, Dtr

from test_nn_and_optim import train_epoch_1
import numpy as np
import pytest

_DEVICES = [stk.cpu(), pytest.param(stk.cuda(),
    marks=pytest.mark.skipif(not stk.cuda().enabled(), reason="No GPU"))]

@pytest.mark.parametrize("device", _DEVICES, ids=["cuda", "cpu"])
def test_dtr(device):
    with dtr.enable_dtr(False):
        start = time.perf_counter()
        np.testing.assert_allclose(train_epoch_1(5, 250, stk.optim.Adam, lr=0.01, weight_decay=0.1),
            np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)
        dur1 = time.perf_counter() - start
    
    with dtr.enable_dtr(True):
        mem_requirement = 174780036
        # assign only 10% needed memory
        Dtr.mem_limit = mem_requirement // 10
        np.random.seed(1)
        start = time.perf_counter()
        np.testing.assert_allclose(train_epoch_1(5, 250, stk.optim.Adam, lr=0.01, weight_decay=0.1),
            np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)
        dur2 = time.perf_counter() - start
    
    print("normal vs dtr") 
    print(f"duration: {dur1}, {dur2}")
    assert dur1 < dur2, "execution with dtr shall take longer"

if __name__ == "__main__":
    test_dtr(device="cpu")
    