import sys
sys.path.append('../python')
import stick as stk
import stick.nn as nn
from models import *

device = stk.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_sum = []
    acc_rate_sum = []
    sample_num = len(dataloader.dataset)
    if opt:
        model.train()
        for X, y in dataloader:
            opt.reset_grad()
            h = model(X)
            acc_rate_sum.append(np.sum(h.numpy().argmax(axis=1) == y.numpy()))
            loss = loss_fn(h, y)
            loss_sum.append(loss.numpy())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            h = model(X)
            loss = loss_fn(h, y)
            acc_rate_sum.append(np.sum(h.numpy().argmax(axis=1) == y.numpy()))
            loss_sum.append(loss.numpy())
    return np.sum(acc_rate_sum) / sample_num, np.average(loss_sum)
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=stk.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn, opt)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_sum = []
    acc_rate_sum = []
    nbatch, _ = data.shape
    sample_num = 0
    hidden = None
    if opt:
        model.train()
        for i in range(0, nbatch - 1, seq_len):
            x, y = stk.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
            opt.reset_grad()
            y_pred, hidden = model(x, hidden)
            acc_rate_sum.append(np.sum(y_pred.numpy().argmax(axis=1) == y.numpy()))
            sample_num += y.shape[0]
            loss = loss_fn(y_pred, y)
            loss_sum.append(loss.numpy())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for i in range(0, nbatch - 1, seq_len):
            x, y = stk.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
            y_pred, hidden = model(x, hidden)
            loss = loss_fn(y_pred, y)
            acc_rate_sum.append(np.sum(y_pred.numpy().argmax(axis=1) == y.numpy()))
            sample_num += y.shape[0]
            loss_sum.append(loss.numpy())
    return np.sum(np.array(acc_rate_sum)) / sample_num, np.average(np.array(loss_sum))
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=stk.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), opt, clip, device, dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = stk.cpu()
    #dataset = stk.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = stk.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=stk.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = stk.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = stk.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
