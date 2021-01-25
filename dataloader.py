import math
import numpy as np


class DataGenerator:
    def __init__(self, data, sequence_length=0, batch_size=64, shuffle=False, output_cols=None):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle  # default to False because pytorch-lightning shuffles internally
        self.pad_value = 0.
        self.indexes = []
        if output_cols is None:
            self.output_cols = data[0][1].columns
        else:
            self.output_cols = output_cols
        for si, s in enumerate(data):
            x = s[0]
            tx = x.shape[0]
            xind = 0
            while tx > 0:
                self.indexes.append((si, xind))
                xind += 1
                tx -= 1
        np.random.shuffle(self.indexes)  # always shuffle once

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        index *= self.batch_size
        this_batch_size = self.batch_size if index + self.batch_size < len(self.indexes) else len(self.indexes) - index
        X = np.zeros((this_batch_size, self.sequence_length, self.data[0][0].shape[1]))
        Y = np.zeros((this_batch_size, self.sequence_length, len(self.output_cols)))
        lengths = np.zeros((this_batch_size, 1))
        for i in range(this_batch_size):
            X[:, i, :], Y[:, i, :], lengths[i] = self.__getsingleitem(index + i)
        return X, Y, lengths

    def __getsingleitem(self, index):
        (seq, stride) = self.indexes[index]
        (X, Y, _, _) = self.data[seq]
        Y = Y.loc[:, self.output_cols]
        if stride + self.sequence_length <= X.shape[0]:
            X = X.iloc[stride:stride + self.sequence_length, :].to_numpy(dtype='float64')
            Y = Y.iloc[stride:stride + self.sequence_length, :].to_numpy(dtype='float64')
            return X, Y, self.sequence_length
        else:
            # pad
            X = X.iloc[stride:X.shape[0], :].to_numpy(dtype='float64')
            padX = np.full((self.sequence_length - X.shape[0], X.shape[1]), self.pad_value)
            Y = Y.iloc[stride:Y.shape[0], :].to_numpy(dtype='float64')
            padY = np.full((self.sequence_length - Y.shape[0], Y.shape[1]), self.pad_value)
            return np.concatenate((X, padX), axis=0), np.concatenate((Y, padY), axis=0), X.shape[0]

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
