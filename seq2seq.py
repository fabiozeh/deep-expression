import numpy as np
import pickle
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dataloader as dl

descr = """
This script loads a sequential dataset with score and performance information and uses it to train
a deep artificial neural network for generating onset timing deviation and peak loudness level of
notes from musical pieces.
"""

# Defining the neural network


class Encoder(nn.Module):
    def __init__(self, n_x, vocab_size, embed_size, dropout):
        super(Encoder, self).__init__()

        self.pitchEmbedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.harmonyRhythmProjector = nn.Linear(in_features=n_x - 1, out_features=embed_size)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(2 * embed_size)
        self.rnn = nn.GRU(2 * embed_size, 2 * embed_size, num_layers=1, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(4 * embed_size)

    def forward(self, pitch, score_feats, lengths):

        pitch = self.pitchEmbedding(pitch)
        score_feats = self.harmonyRhythmProjector(score_feats)
        src_vec = torch.cat([pitch, score_feats], dim=2)
        src_vec = self.norm1(self.drop1(src_vec))

        sequence = nn.utils.rnn.pack_padded_sequence(src_vec, lengths.cpu(), enforce_sorted=False)
        output, _ = self.rnn(sequence)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return src_vec, self.norm2(self.drop2(output))


class Decoder(nn.Module):
    def __init__(self, n_y, hidden_size, enc_hidden_size, dropout=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.y_proj = nn.Linear(n_y, hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_size, 1, kdim=enc_hidden_size, vdim=enc_hidden_size)
        self.rnn = nn.GRU(2 * hidden_size, hidden_size, num_layers=1)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.ff1 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.drop3 = nn.Dropout(dropout)
        self.ff2 = nn.Linear(2 * hidden_size, n_y, bias=False)

    def forward(self, x_vec, y_prev, encoder_out, dec_hidden):
        """
        Generates outputs for a single step in the sequence (one note)
        """
        y_projected = self.drop1(self.y_proj(y_prev))  # (1, batch_size, hidden_size)

        # shapes: (1, b, h) <- ( (1, b, h), (len_seq, b, enc_hidden), (len_seq, b, enc_hidden) )
        context, _ = self.attention(dec_hidden, encoder_out, encoder_out)
        rnn_out, new_dec_hidden = self.rnn(torch.cat([y_projected, context], dim=2), dec_hidden)

        rnn_out = self.norm(self.drop2(rnn_out))
        out = self.ff2(self.drop3(F.relu(self.ff1(torch.cat([rnn_out, y_projected, context, x_vec], dim=2)))))
        return out, new_dec_hidden

class Net(pl.LightningModule):

    def __init__(self, n_x, n_y, vocab_size, hidden_size=64, dropout_rate=0.1, lr=1e-4,
                 context=0, window=0, scheduler_step=4, lr_decay_by=0.25):
        super(Net, self).__init__()

        assert hidden_size % 2 == 0, "hidden_size must be multiple of 2"

        self.save_hyperparameters()

        self.rng = np.random.default_rng()

        self.encoder = Encoder(n_x, vocab_size, int(hidden_size / 2), dropout_rate)
        self.decoder = Decoder(n_y, hidden_size, 2 * hidden_size, dropout_rate)

    def forward(self, pitch, score_feats, lengths):
        """
        Generate the entire sequence
        """
        src_vec, encoded_score = self.encoder(pitch, score_feats, lengths)
        hidden = torch.zeros((1, pitch.shape[1], self.hparams.hidden_size), device=self.device)
        y = torch.zeros((pitch.shape[0], pitch.shape[1], self.hparams.n_y), device=self.device)
        prev_y = torch.zeros((1, pitch.shape[1], self.hparams.n_y), device=self.device)
        for i in range(pitch.shape[0]):
            prev_y, hidden = self.decoder(src_vec[i, :, :].unsqueeze(0), prev_y, encoded_score, hidden)
            y[i, :, :] = prev_y
        return y

    def training_step(self, batch, batch_idx):
        """
        This method doesn't use self.forward directly so we can apply teacher forcing
        on a fraction of the steps.
        """
        pitch, score_feats, y, lengths = batch
        if len(pitch.shape) < 2:
            pitch = pitch.unsqueeze(1)
            score_feats = score_feats.unsqueeze(1)
            y = y.unsqueeze(1)

        # encode x (score)
        src_vec, encoded_score = self.encoder(pitch, score_feats, lengths)

        # iterate generating y
        teacher_forcing_ratio = 0.5

        hidden = torch.zeros((1, score_feats.shape[1], self.hparams.hidden_size), device=self.device)
        y_hat = torch.zeros((y.shape[0], y.shape[1], self.hparams.n_y), device=self.device)
        prev_y = torch.zeros((1, score_feats.shape[1], self.hparams.n_y), device=self.device)
        for i in range(pitch.shape[0]):
            prev_y, hidden = self.decoder(src_vec[i, :, :].unsqueeze(0), prev_y, encoded_score, hidden)
            y_hat[i, :, :] = prev_y
            if self.rng.random() > teacher_forcing_ratio:
                prev_y = y[i, :, :].view(1, -1, self.hparams.n_y)

        if self.hparams.window:
            ctx = self.hparams.context
            if not ctx:
                ctx = y_hat.shape[0] - self.hparams.window
            y_hat = y_hat[ctx:ctx + self.hparams.window, :, :]
            y = y[ctx:ctx + self.hparams.window, :, :]

        loss = F.mse_loss(y_hat, y)
        if (batch_idx + 1) % 500 == 0:
            self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pitch, score_feats, y, lengths = batch
        if len(pitch.shape) < 2:
            pitch = pitch.unsqueeze(1)
            score_feats = score_feats.unsqueeze(1)
            y = y.unsqueeze(1)

        y_hat = self.forward(pitch, score_feats, lengths)

        if self.hparams.window:
            ctx = self.hparams.context
            if not ctx:
                ctx = y_hat.shape[0] - self.hparams.window
            y_hat = y_hat[ctx:ctx + self.hparams.window, :, :]
            y = y[ctx:ctx + self.hparams.window, :, :]

        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.scheduler_step, gamma=self.hparams.lr_decay_by)
        return [optimizer], [scheduler]


def evaluation(sequences, sequence_length, model, output_cols, stride=0, context=0, pad_both_ends=False):
    loader = dl.ValidationDataset(sequences,
                                  vocab_col=sequences[0][0][0].columns.get_loc("pitch"),
                                  sequence_length=sequence_length,
                                  output_cols=output_cols,
                                  stride=stride,
                                  context=context,
                                  pad_both_ends=pad_both_ends,
                                  device=model.device)
    Y_hat = []
    for piece in range(len(loader)):
        (pch, s_f, Y, lth) = loader[piece]
        out = model(pch, s_f, lth)
        out = out.detach().numpy()
        y_hat_p = np.zeros((sequences[piece][0][1].shape[0], len(output_cols)))
        ind = 0
        for s in range(out.shape[1] - 1):
            y_hat_p[ind:ind + stride, :] = out[context:context + stride, s, :]
            ind += stride
        y_hat_p[ind:, :] = out[context:context + y_hat_p.shape[0] - ind, -1, :]
        Y_hat.append(y_hat_p)

    mse = np.zeros((len(sequences), Y_hat[0].shape[1]))
    for i, S in enumerate(sequences):
        Y = S[0][1]
        Y = Y.loc[:, output_cols]
        mse[i, :] = np.mean((Y_hat[i][:Y.shape[0], :] - Y) ** 2) / np.mean(Y ** 2)
    return Y_hat, mse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data', help='data file for training or evaluation.')
    parser.add_argument('--val-data', help='validation set file')
    parser.add_argument('-m', '--model-state', default='seq2seq_model_state.pth',
                        help='PyTorch state dictionary file name for saving (on train mode) or loading (on eval mode).')
    parser.add_argument('--eval', action='store_true', help='runs the script for evaluation.')
    parser.add_argument('-g', '--gen-attr', nargs='+',
                        choices=('ioiRatio', 'peakLevel', 'localTempo', 'timingDev', 'timingDevLocal', 'durationSecs', 'velocity'),
                        default=['ioiRatio', 'peakLevel', 'durationSecs'],
                        help='expressive attributes to learn/generate.')
    parser.add_argument('--vocab-size', type=int, default=85, help='pitch vocabulary size.')
    parser.add_argument('-r', '--lr', type=float, default=1e-5, help='learning rate for training.')
    parser.add_argument('-l', '--seq-len', type=int, default=32, help='number of notes read by model at once.')
    parser.add_argument('-s', '--hidden-size', type=int, default=64, help='size of hidden model layers.')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='model dropout rate.')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='mini-batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of training epochs.')
    parser.add_argument('--scheduler-step', type=int, default=4, help='epochs between lr decays.')
    parser.add_argument('--lr-decay-by', type=float, default=0.25, help='lr decay rate on scheduler steps.')
    parser.add_argument('--stride', type=int, default=24, help='the stride in the notes sliding window.')
    parser.add_argument('--context', type=int, default=4, help='no. of notes ignored at window start.')
    parser.add_argument('--no-ctx-train', action='store_true', help='ignore context when training.')
    parser.add_argument('--dev-run', action='store_true', help='run script for testing purposes.')
    parser.add_argument('-w', '--workers', type=int, default=8, help='workers for the lightning trainer.')
    parser.add_argument('--cpu-only', action='store_true', help="don't train on GPUs.")

    args = parser.parse_args()

    #  Loading data
    with open(args.data, 'rb') as data_file:
        train = pickle.load(data_file)
    if args.eval:
        val = train
    if args.val_data:
        with open(args.val_data, 'rb') as val_data_file:
            val = pickle.load(val_data_file)

    # Instantiating model

    # pl.seed_everything(1728) # TODO This alone doesn't seem to work

    model = Net(train[0][0][0].shape[1],
                len(args.gen_attr),
                vocab_size=args.vocab_size,  # vocab size=81 + ix: 0 = pad, len+1 = UKN, len+2 = END, len+3 = SOS
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                lr=args.lr,
                context=(0 if args.no_ctx_train else args.context),
                window=(0 if args.no_ctx_train else args.stride),
                scheduler_step=args.scheduler_step,
                lr_decay_by=args.lr_decay_by)

    # Training model

    if not args.eval:

        print("Beginning sequence to sequence model training invoked with command:")
        print(' '.join(sys.argv))

        if args.cpu_only:
            trainer = pl.Trainer(fast_dev_run=args.dev_run,
                                 progress_bar_refresh_rate=20, max_epochs=args.epochs,
                                 val_check_interval=0.25)
        else:
            trainer = pl.Trainer(gpus=-1, accelerator='ddp', fast_dev_run=args.dev_run,
                                 progress_bar_refresh_rate=20, max_epochs=args.epochs,
                                 val_check_interval=0.25)

        if args.seq_len == 0:
            train_ds = dl.FullPieceDataset(train,
                                           vocab_col=train[0][0][0].columns.get_loc("pitch"),
                                           output_cols=args.gen_attr)
            if val:
                val_ds = dl.FullPieceDataset(val,
                                             vocab_col=val[0][0][0].columns.get_loc("pitch"),
                                             output_cols=args.gen_attr)
        else:
            train_ds = dl.TrainDataset(train,
                                       vocab_col=train[0][0][0].columns.get_loc("pitch"),
                                       sequence_length=args.seq_len,
                                       output_cols=args.gen_attr,
                                       context=args.context,
                                       dummy=args.dev_run)
            if val:
                val_ds = dl.ValidationDataset(val,
                                              vocab_col=val[0][0][0].columns.get_loc("pitch"),
                                              sequence_length=args.seq_len,
                                              output_cols=args.gen_attr,
                                              stride=args.stride,
                                              context=args.context,
                                              pad_both_ends=(not args.no_ctx_train),
                                              device=model.device)
        if val:
            trainer.fit(model,
                        DataLoader(train_ds,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   shuffle=True,
                                   collate_fn=dl.TrainDataset.collate_fn),
                        DataLoader(val_ds,
                                   batch_size=None,
                                   num_workers=args.workers))

        # Saving model
        torch.save(model.state_dict(), args.model_state)

    else:
        # Load model
        model.load_state_dict(torch.load(args.model_state))
        model.eval()

        _, mse = evaluation(val, args.seq_len, model, output_cols=args.gen_attr,
                            stride=args.stride, context=args.context,
                            pad_both_ends=(not args.no_ctx_train))

        for i, col in enumerate(args.gen_attr):
            print('Validation set MSE for ' + col + ': ' + str(np.mean(mse[:, i])))
