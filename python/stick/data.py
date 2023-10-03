import numpy as np
from .ops import Tensor
import os, gzip, struct, pickle
from typing import Optional, List


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        img_pad = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        H, W, _ = img.shape
        return img_pad[shift_x + self.padding : H + shift_x + self.padding, shift_y + self.padding : W + shift_y + self.padding, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = -1
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        samples = [self.dataset[i] for i in self.ordering[self.idx]]
        return [Tensor([samples[i][j] for i in range(len(samples))]) for j in range(len(samples[0]))]
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as img_file:
            magic_number, num_examples, row, column = struct.unpack('>4I', img_file.read(16))
            assert(magic_number == 2051)
            input_dim = row * column
            X = np.array(struct.unpack(str(input_dim * num_examples) + 'B', img_file.read()), dtype=np.float32).reshape(num_examples, input_dim)
            X -= np.min(X)
            X /= np.max(X)
        with gzip.open(label_filename, 'rb') as label_file:
            magic_number, num_items = struct.unpack('>2I', label_file.read(8))
            assert(magic_number == 2049)
            y = np.array(struct.unpack(str(num_items) + 'B', label_file.read()), dtype=np.uint8)
        self.images = X
        self.img_row = row
        self.img_column = column
        self.labels = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        labels = self.labels[index]
        if len(imgs.shape) > 1:
            imgs = np.array([self.apply_transforms(img.reshape(self.img_row, self.img_column, 1)).reshape(imgs[0].shape) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(self.img_row, self.img_column, 1)).reshape(imgs.shape)
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        X = list()
        y = list()
        if train:
            for i in range(1, 6):
                with open(f"{base_folder}/data_batch_{i}", "rb") as fo:
                    dict = pickle.load(fo, encoding="bytes")
                    train_batch, train_labels = dict[b"data"].astype(float), dict[b"labels"]
                    X.append(train_batch)
                    y.append(train_labels)
        else:
            with open(f"{base_folder}/test_batch", "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
                test_batch, test_labels = dict[b"data"].astype(float), dict[b"labels"]
                X.append(test_batch)
                y.append(test_labels)
        X = np.concatenate(X, axis=0)
        X = X.reshape((X.shape[0], 3, 32, 32)) / 255
        y = np.concatenate(y)
        self.images = X
        self.labels = y
        self.p = p
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        ids = []
        for line in lines:
            if i == max_lines:
                break
            words = line.strip().split(' ')
            for word in words:
                ids.append(self.dictionary.add_word(word))
            ids.append(self.dictionary.add_word('<eos>'))
            i += 1
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    data = np.array(data, dtype=dtype)
    nbatch = len(data) // batch_size
    data = data[:batch_size*nbatch].reshape((batch_size, nbatch)).T
    return data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    # NOTE: make sure `data` and `target` have same 
    # seq_len when `i` nearly reach the end of `batches`
    bptt = min(bptt, len(batches)-1-i)
    data = batches[i:i+bptt]
    target = batches[i+1:i+1+bptt]
    data = Tensor(data, device=device, dtype=dtype)
    target = Tensor(target.reshape(-1), device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION
