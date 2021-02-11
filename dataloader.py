import torch
import numpy as np


def pitchVocabularyFmt(X, vocab_col):
    """
    Produces the tensors for training with a pitch vocabulary encoding.
    """
    pitch = torch.LongTensor(X[:, vocab_col])
    score_feats = torch.cat([torch.FloatTensor(X[:, :vocab_col]),
                            torch.FloatTensor(X[:, vocab_col + 1:])], dim=1)
    return pitch, score_feats


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab_col, sequence_length, output_cols=None, dummy=False):
        self.data = data
        self.sequence_length = sequence_length
        self.vocab_col = vocab_col
        self.dummy = dummy
        self.indexes = []
        if output_cols is None:
            self.output_cols = data[0][0][1].columns
        else:
            self.output_cols = output_cols
        for si, s in enumerate(data):
            x = s[0][0]
            tx = x.shape[0]
            xind = 0
            self.indexes.append((si, xind))
            while tx > 0:
                xind += 1
                tx -= 1
                self.indexes.append((si, xind))
        np.random.shuffle(self.indexes)  # always shuffle once

    def __len__(self):
        if self.dummy:
            return 32
        else:
            return len(self.indexes)

    def __getitem__(self, index):
        (seq, offset) = self.indexes[index]
        (X, Y, _) = self.data[seq][0]
        Y = Y.loc[:, self.output_cols]
        if offset + self.sequence_length <= X.shape[0]:
            X = X.iloc[offset:offset + self.sequence_length, :].to_numpy(dtype='float64')
            Y = Y.iloc[offset:offset + self.sequence_length, :].to_numpy(dtype='float64')
        else:
            X = X.iloc[offset:, :].to_numpy(dtype='float64')
            Y = Y.iloc[offset:, :].to_numpy(dtype='float64')

        pitch, score_feats = pitchVocabularyFmt(X, self.vocab_col)
        return pitch, score_feats, torch.FloatTensor(Y), X.shape[0]

    @staticmethod
    def collate_fn(batch):
        pitch, score_feats, Y, length = (
            [element[0] for element in batch],
            [element[1] for element in batch],
            [element[2] for element in batch],
            [element[3] for element in batch],
        )

        # Pad batch
        pitch = torch.nn.utils.rnn.pad_sequence(pitch, batch_first=False)
        score_feats = torch.nn.utils.rnn.pad_sequence(score_feats, batch_first=False)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=False)

        return pitch, score_feats, Y, torch.LongTensor(length).view(-1)


class ValidationDataset(torch.utils.data.Dataset):
    """
    This dataset splits pieces according to sequence length to form one batch
    per piece and sliding the input window according to stride.
    """
    def __init__(self, data, vocab_col, sequence_length, output_cols=None, stride=0, device=None):
        self.data = data
        self.sequence_length = sequence_length
        self.vocab_col = vocab_col
        self.indexes = []
        if stride == 0:
            self.stride = self.sequence_length
        else:
            self.stride = stride
        self.device = device

        assert self.stride <= self.sequence_length, "Invalid combination of stride and sequence_length"

        if output_cols is None:
            self.output_cols = data[0][0][1].columns
        else:
            self.output_cols = output_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (X, Y, _) = self.data[index][0]
        Y = Y.loc[:, self.output_cols]
        if X.shape[0] % self.stride == 0:
            batch_size = int(X.shape[0] / self.stride)
        else:
            batch_size = int(X.shape[0] / self.stride + 1)

        pitch_batch = np.zeros((self.sequence_length, batch_size))
        score_feats_batch = np.zeros((self.sequence_length, batch_size, X.shape[1] - 1))
        Y_batch = np.zeros((self.sequence_length, batch_size, len(self.output_cols)))
        length_batch = np.zeros(batch_size)

        xind = 0
        for i in range(batch_size):
            if xind + self.sequence_length <= X.shape[0]:
                pitch_batch[:, i] = X.iloc[xind:xind + self.sequence_length, self.vocab_col].to_numpy(dtype='int64')
                score_feats_batch[:, i, :self.vocab_col] = X.iloc[xind:xind + self.sequence_length, :self.vocab_col].to_numpy(dtype='float64')
                score_feats_batch[:, i, self.vocab_col:] = X.iloc[xind:xind + self.sequence_length, self.vocab_col + 1:].to_numpy(dtype='float64')
                Y_batch[:, i, :] = Y.iloc[xind:xind + self.sequence_length, :].to_numpy(dtype='float64')
                length_batch[i] = self.sequence_length
            else:
                sz_last = X.shape[0] - xind
                pitch_batch[:sz_last, i] = X.iloc[xind:, self.vocab_col].to_numpy(dtype='int64')
                score_feats_batch[:sz_last, i, :self.vocab_col] = X.iloc[xind:, :self.vocab_col].to_numpy(dtype='float64')
                score_feats_batch[:sz_last, i, self.vocab_col:] = X.iloc[xind:, self.vocab_col + 1:].to_numpy(dtype='float64')
                Y_batch[:sz_last, i, :] = Y.iloc[xind:, :].to_numpy(dtype='float64')
                length_batch[i] = sz_last
            xind += self.stride

        if self.device is not None:
            pitch_batch = torch.LongTensor(pitch_batch, device=self.device)
            score_feats_batch = torch.FloatTensor(score_feats_batch, device=self.device)
            Y_batch = torch.FloatTensor(Y_batch, device=self.device)
            length_batch = torch.LongTensor(length_batch, device=self.device)
        else:
            pitch_batch = torch.LongTensor(pitch_batch)
            score_feats_batch = torch.FloatTensor(score_feats_batch)
            Y_batch = torch.FloatTensor(Y_batch)
            length_batch = torch.LongTensor(length_batch)

        return pitch_batch, score_feats_batch, Y_batch, length_batch
