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
    """
    This class prepares the provided data for training in a sequence to sequence model.
    Pieces are split in shorter sequences beginning in each existing note for maximum
    data augmentation.
    _data_ should be in the format stored by the 'sequence preparation' notebook.
    _vocab_col_ is the index of the column corresponding to the lexical vocabulary of the
        dependent variable pandas array.
    _sequence_length_ is the integer length of sequences to be passed to the model.
    _output_cols_ is a subset of the independent variable array column names provided as
        a string list. Default: all columns
    _context_ is an integer number of sequence steps that will be read by the model to
        provide context for predictions, so no prediction will be generated for them.
        This class will pad the start of pieces so the model can also learn to predict
        the first _context_ notes of a piece. Default: no context (don't pad).
    _dummy_ if True will restrict the length to 32 instances for testing.
    """
    def __init__(self, data, vocab_col, sequence_length, output_cols=None, context=0, dummy=False):
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
            tx = x.shape[0] + context
            xind = -context
            self.indexes.append((si, xind))
            while tx > 1:
                xind += 1
                tx -= 1
                self.indexes.append((si, xind))
        np.random.seed(777)
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
        print(str(seq) + '  ' + str(offset))
        print(X.shape[0])
        if offset < 0:
            if offset + self.sequence_length <= X.shape[0]:
                Xp = np.zeros((self.sequence_length, X.shape[1]), dtype='float64')
                Yp = np.zeros((self.sequence_length, Y.shape[1]), dtype='float64')
                Xp[-offset:, :] = X.iloc[:offset + self.sequence_length, :].to_numpy(dtype='float64')
                Yp[-offset:, :] = Y.iloc[:offset + self.sequence_length, :].to_numpy(dtype='float64')
                X = Xp
                Y = Yp
            else:
                Xp = np.zeros((X.shape[0] - offset, X.shape[1]), dtype='float64')
                Yp = np.zeros((X.shape[0] - offset, Y.shape[1]), dtype='float64')
                Xp[-offset:, :] = X.to_numpy(dtype='float64')
                Yp[-offset:, :] = Y.to_numpy(dtype='float64')
                X = Xp
                Y = Yp
        elif offset + self.sequence_length <= X.shape[0]:
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
    def __init__(self, data, vocab_col, sequence_length, output_cols=None, stride=0, context=0, pad_both_ends=False, device=None):
        self.data = data
        self.sequence_length = sequence_length
        self.vocab_col = vocab_col
        self.context = context
        self.pad_both_ends = pad_both_ends
        if stride == 0:
            self.stride = self.sequence_length
        else:
            self.stride = stride
        self.device = device

        assert self.stride + self.context <= self.sequence_length, "Invalid combination of stride, context and sequence_length"

        if output_cols is None:
            self.output_cols = data[0][0][1].columns
        else:
            self.output_cols = output_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (X, Y, _) = self.data[index][0]
        Y = Y.loc[:, self.output_cols]
        if self.pad_both_ends:
            if (X.shape[0] + self.context) % self.stride == 0:
                batch_size = int(X.shape[0] / self.stride)
            else:
                batch_size = int(X.shape[0] / self.stride + 1)
        else:
            if (X.shape[0] - self.sequence_length + self.context) % self.stride == 0:
                batch_size = int((X.shape[0] - self.sequence_length + self.context) / self.stride + 1)
            else:
                batch_size = int((X.shape[0] - self.sequence_length + self.context) / self.stride + 2)

        pitch_batch = np.zeros((self.sequence_length, batch_size))
        score_feats_batch = np.zeros((self.sequence_length, batch_size, X.shape[1] - 1))
        Y_batch = np.zeros((self.sequence_length, batch_size, len(self.output_cols)))
        length_batch = np.zeros(batch_size)

        pitch_batch[self.context:, 0] = X.iloc[:self.sequence_length - self.context, self.vocab_col].to_numpy(dtype='int64')
        score_feats_batch[self.context:, 0, :self.vocab_col] = X.iloc[:self.sequence_length - self.context, :self.vocab_col].to_numpy(dtype='float64')
        score_feats_batch[self.context:, 0, self.vocab_col:] = X.iloc[:self.sequence_length - self.context, self.vocab_col + 1:].to_numpy(dtype='float64')
        Y_batch[self.context:, 0, :] = Y.iloc[:self.sequence_length - self.context, :].to_numpy(dtype='float64')
        length_batch[0] = self.sequence_length

        xind = self.stride - self.context
        for i in range(1, batch_size):
            if xind < 0:
                if xind + self.sequence_length <= X.shape[0]:
                    pitch_batch[-xind:, i] = X.iloc[:xind + self.sequence_length, self.vocab_col].to_numpy(dtype='int64')
                    score_feats_batch[-xind:, i, :self.vocab_col] = X.iloc[:xind + self.sequence_length, :self.vocab_col].to_numpy(dtype='float64')
                    score_feats_batch[-xind:, i, self.vocab_col:] = X.iloc[:xind + self.sequence_length, self.vocab_col + 1:].to_numpy(dtype='float64')
                    Y_batch[-xind:, i, :] = Y.iloc[:xind + self.sequence_length, :].to_numpy(dtype='float64')
                    length_batch[i] = self.sequence_length
                else:
                    sz_last = X.shape[0] - xind
                    pitch_batch[-xind:sz_last, i] = X.iloc[:, self.vocab_col].to_numpy(dtype='int64')
                    score_feats_batch[-xind:sz_last, i, :self.vocab_col] = X.iloc[:, :self.vocab_col].to_numpy(dtype='float64')
                    score_feats_batch[-xind:sz_last, i, self.vocab_col:] = X.iloc[:, self.vocab_col + 1:].to_numpy(dtype='float64')
                    Y_batch[-xind:sz_last, i, :] = Y.to_numpy(dtype='float64')
                    length_batch[i] = sz_last
            elif xind + self.sequence_length <= X.shape[0]:
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
        else:
            pitch_batch = torch.LongTensor(pitch_batch)
            score_feats_batch = torch.FloatTensor(score_feats_batch)
            Y_batch = torch.FloatTensor(Y_batch)

        return pitch_batch, score_feats_batch, Y_batch, torch.LongTensor(length_batch)


class FullPieceDataset(torch.utils.data.Dataset):
    """
    This dataset returns entire pieces instead of a maximum length of notes.
    """
    def __init__(self, data, vocab_col, output_cols=None):
        self.data = data
        self.vocab_col = vocab_col

        if output_cols is None:
            self.output_cols = data[0][0][1].columns
        else:
            self.output_cols = output_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (X, Y, _) = self.data[index][0]
        X = X.to_numpy(dtype='float64')
        Y = Y.loc[:, self.output_cols].to_numpy(dtype='float64')
        length = X.shape[0]

        pitch, score_feats = pitchVocabularyFmt(X, self.vocab_col)

        return pitch, score_feats, torch.FloatTensor(Y), torch.LongTensor([length]).view(-1)
