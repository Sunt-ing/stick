import sys
sys.path.append('../python')
import stick as stk
import stick.nn as nn
import numpy as np

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1, device=None):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(nn.Linear(dim, hidden_dim, device=device), norm(hidden_dim), nn.ReLU(), 
                        nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim, device=device), norm(dim))
    return nn.Sequential(nn.Residual(seq), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1, device=None):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim, device=device)
    rbs = [ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob, device=device) for _ in range(num_blocks)]
    linear2 = nn.Linear(hidden_dim, num_classes, device=device)
    return nn.Sequential(linear1, nn.ReLU(), *rbs, linear2)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    err_rates = list()
    losses = list()
    softmax_loss = nn.SoftmaxLoss()
    sample_num = len(dataloader.dataset)
    if opt:
        model.train()
        for X, y in dataloader:
            opt.reset_grad()
            h = model(X)
            loss = softmax_loss(h, y)
            err_rates.append(np.sum(h.numpy().argmax(axis=1) != y.numpy()))
            losses.append(loss.numpy())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            h = model(X)
            loss = softmax_loss(h, y)
            err_rates.append(np.sum(h.numpy().argmax(axis=1) != y.numpy()))
            losses.append(loss.numpy())
    return np.sum(err_rates) / sample_num, np.average(losses)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=stk.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = stk.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = stk.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = stk.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = stk.data.DataLoader(test_set, batch_size, shuffle=False)
    model = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    opti = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt=opti)
    test_err, test_loss = epoch(test_loader, model, opt=None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
