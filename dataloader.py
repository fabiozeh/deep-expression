import torch


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data, vocab_col, sequence_length, output_cols=None):
        self.data = data
        self.sequence_length = sequence_length
        self.vocab_col = vocab_col
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
        # np.random.shuffle(self.indexes)  # always shuffle once

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        (seq, stride) = self.indexes[index]
        (X, Y, _) = self.data[seq][0]
        Y = Y.loc[:, self.output_cols]
        if stride + self.sequence_length <= X.shape[0]:
            X = X.iloc[stride:stride + self.sequence_length, :].to_numpy(dtype='float64')
            Y = Y.iloc[stride:stride + self.sequence_length, :].to_numpy(dtype='float64')
            length = self.sequence_length
        else:
            length = X.shape[0] - stride
            X = X.iloc[stride:, :].to_numpy(dtype='float64')
            Y = Y.iloc[stride:, :].to_numpy(dtype='float64')

        pitch = torch.LongTensor(X[:, self.vocab_col])
        harmRhythm = torch.cat([torch.FloatTensor(X[:, :self.vocab_col]),
                                torch.FloatTensor(X[:, self.vocab_col + 1:])], dim=1)
        return pitch, harmRhythm, torch.FloatTensor(Y), length

    @staticmethod
    def collate_fn(batch):
        pitch, harmRhythm, Y, length = (
            [element[0] for element in batch],
            [element[1] for element in batch],
            [element[2] for element in batch],
            [element[3] for element in batch],
        )

        # Pad batch
        pitch = torch.nn.utils.rnn.pad_sequence(pitch, batch_first=False)
        harmRhythm = torch.nn.utils.rnn.pad_sequence(harmRhythm, batch_first=False)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=False)

        return pitch, harmRhythm, Y, torch.LongTensor(length).view(-1)
