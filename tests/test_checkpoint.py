import sys
sys.path.append("./python")
import stick as stk
from stick import checkpoint, nn
import numpy as np
from stick import backend_ndarray as nd
import time, pytest


# a ResNet built on TestModule (Sequential or Memonger)
def MLPResNet(TestModule, dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1, device=None):
    def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
        seq = TestModule(nn.Linear(dim, hidden_dim, device=device), norm(hidden_dim, device=device), 
                         nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim, device=device), 
                         norm(dim, device=device))
        return TestModule(nn.Residual(seq), nn.ReLU())

    linear1 = nn.Linear(dim, hidden_dim, device=device)
    rbs = [ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]
    linear2 = nn.Linear(hidden_dim, num_classes, device=device)
    return TestModule(linear1, nn.ReLU(), *rbs, linear2)


def model_res(TestModule, device):
    begin = stk.array_api.NDARRAY_COUNTER
    np.random.seed(233)
    model = MLPResNet(TestModule, 28 * 28, drop_prob=0, device=device)
    # 1 is batch size
    inputs = stk.randn(1, 28 * 28, requires_grad=True, device=device)
    # set state
    model.train()
    h = model(inputs)
    
    # --- backward code ---
    # y = stk.randn(2, requires_grad=True)
    # loss = nn.SoftmaxLoss()(h, y)
    # loss.backward()
    # out_grad = stk.init.ones(*loss.shape, dtype=loss.dtype, device=loss.device)
    # stk.ops.compute_gradient_of_variables(loss, out_grad)

    ret = h.numpy()
    cnt = stk.array_api.NDARRAY_COUNTER - begin
    return ret, cnt


_DEVICES = [pytest.param(stk.cuda(),
    marks=pytest.mark.skipif(not stk.cuda().enabled(), reason="No GPU"))]


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_checkpoint(device):
    start = time.perf_counter()
    output1, cnt1 = model_res(nn.Sequential, device)
    mid = time.perf_counter()
    output2, cnt2 = model_res(checkpoint.Memonger, device)
    end = time.perf_counter()
    dur1 = "{:.5f}".format(mid - start)
    dur2 = "{:.5f}".format(end - mid)
    
    print("Sequential vs Memonger")
    print(f"duration: {dur1}, {dur2}")
    print(f"NDArray num: {cnt1}, {cnt2}")

    np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-5)
    assert cnt1 > cnt2

if __name__ == "__main__":
    test_checkpoint()
