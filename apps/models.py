# Reference: 
# https://github.com/Somoku/CMU10-714-DLSys
# https://github.com/YuanchengFang/dlsys_solution

import sys
sys.path.append('./python')
import stick as stk
import stick.nn as nn
import numpy as np


np.random.seed(0)

def ConvBNBlock(in_channels, out_channels, kernel_size, stride, device=None):
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    return nn.Sequential(nn.Conv(in_channels, out_channels, kernel_size, stride, device=device),
                         nn.BatchNorm2d(out_channels, device=device),
                         nn.ReLU())

class ResNet9(stk.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(ConvBNBlock(3, 16, 7, 4, device=device),
                                   ConvBNBlock(16, 32, 3, 2, device=device),
                                   nn.Residual(nn.Sequential(ConvBNBlock(32, 32, 3, 1, device=device),
                                                             ConvBNBlock(32, 32, 3, 1, device=device))),
                                   ConvBNBlock(32, 64, 3, 2, device=device),
                                   ConvBNBlock(64, 128, 3, 2, device=device),
                                   nn.Residual(nn.Sequential(ConvBNBlock(128, 128, 3, 1, device=device),
                                                             ConvBNBlock(128, 128, 3, 1, device=device))),
                                   nn.Flatten(),
                                   nn.Linear(128, 128, device=device),
                                   nn.ReLU(),
                                   nn.Linear(128, 10, device=device))
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size, device, dtype)
        if seq_model == 'rnn':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise NotImplementedError()
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x = self.embedding(x)   # (seq_len, bs, embedding_size)
        x, h_ = self.model(x, h) # (seq_len, bs, hidden_size)    
        x = self.linear(x.reshape((seq_len*bs, self.hidden_size)))  
        return x, h_
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = stk.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = stk.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = stk.data.DataLoader(cifar10_train_dataset, 128, stk.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
