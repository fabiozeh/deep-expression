import numpy as np
import pickle
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dataloader as dl
import seq2seq as s2s

descr = """
This script tries transfer learning from the maestro dataset to the beethoven sonatas.
"""

# Defining the neural network


class InstrumentEncoder(nn.Module):
    def __init__(self, n_x, embed_size, dropout):
        super(InstrumentEncoder, self).__init__()

        self.instff = nn.Linear(n_x, embed_size)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(2 * embed_size)
        self.ff = nn.Linear(3 * embed_size, 2 * embed_size)
        self.ffnorm = nn.LayerNorm(2 * embed_size)

    def forward(self, pitch, score_feats, lengths, instrument, encoder):

        pitch = encoder.pitchEmbedding(pitch)
        score_feats = encoder.harmonyRhythmProjector(score_feats)
        src_vec = torch.cat([pitch, score_feats], dim=2)
        src_vec = encoder.ffnorm(encoder.drop1(encoder.ff(encoder.norm1(F.relu(src_vec)))))

        proj_inst = self.instff(instrument)
        all_feats = torch.cat([src_vec, proj_inst], dim=2)
        all_feats = self.ffnorm(self.drop1(self.ff(self.norm1(F.relu(all_feats)))))

        sequence = nn.utils.rnn.pack_padded_sequence(all_feats, lengths.cpu(), enforce_sorted=False)
        output, _ = encoder.rnn(sequence)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return src_vec, encoder.norm2(encoder.drop2(output))


class Net(pl.LightningModule):

    def __init__(self, n_y, vocab_size, hidden_size=64, dropout_rate=0.1, lr=1e-4,
                 context=0, window=0, scheduler_step=10000, lr_decay_by=0.9, dec_layers=1, enc_layers=1):
        super(Net, self).__init__()

        assert hidden_size % 2 == 0, "hidden_size must be multiple of 2"

        self.save_hyperparameters()

        self.rng = np.random.default_rng()

        self.encoder = s2s.Encoder(2, vocab_size, int(hidden_size / 2), dropout_rate, gru_layers=enc_layers)
        self.inst_encoder = InstrumentEncoder(2, int(hidden_size / 2), dropout_rate)
        self.decoder = s2s.Decoder(n_y, hidden_size, 2 * hidden_size, dropout_rate, gru_layers=dec_layers)

    def forward(self, pitch, score_feats, instrument, lengths):
        """
        Generate the entire sequence
        """
        src_vec, encoded_score = self.inst_encoder(pitch, score_feats, lengths, instrument, self.encoder)
        hidden = torch.zeros((self.hparams.dec_layers, pitch.shape[1], self.hparams.hidden_size), device=self.device)
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
        pitch, score_feats, instrument, y, lengths = batch
        if len(pitch.shape) < 2:
            pitch = pitch.unsqueeze(1)
            score_feats = score_feats.unsqueeze(1)
            instrument = instrument.unsqueeze(1)
            y = y.unsqueeze(1)

        # encode x (score)
        src_vec, encoded_score = self.inst_encoder(pitch, score_feats, lengths, instrument, self.encoder)

        # iterate generating y
        teacher_forcing_ratio = 0.5

        hidden = torch.zeros((self.hparams.dec_layers, score_feats.shape[1], self.hparams.hidden_size), device=self.device)
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
        pitch, score_feats, instrument, y, lengths = batch
        if len(pitch.shape) < 2:
            pitch = pitch.unsqueeze(1)
            score_feats = score_feats.unsqueeze(1)
            instrument = instrument.unsqueeze(1)
            y = y.unsqueeze(1)

        y_hat = self.forward(pitch, score_feats, instrument, lengths)

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
        scheduler = {'scheduler': optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=self.hparams.scheduler_step,
                                                            gamma=self.hparams.lr_decay_by),
                     'interval': 'step'}
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
        if pad_both_ends:
            ind = 0
        else:
            y_hat_p[:context, :] = out[:context, 0, :]
            ind = context
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


def run_with_args(model, train, val, args):
    if not args.eval:

        print("Beginning sequence to sequence model training invoked with command:")
        print(' '.join(sys.argv))

        if args.cpu_only:
            trainer = pl.Trainer(fast_dev_run=args.dev_run,
                                 progress_bar_refresh_rate=100, max_epochs=args.epochs,
                                 max_steps=args.max_steps, val_check_interval=500)
        else:
            trainer = pl.Trainer(gpus=-1, accelerator='ddp', fast_dev_run=args.dev_run,
                                 plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                                 progress_bar_refresh_rate=100, max_epochs=args.epochs,
                                 max_steps=args.max_steps, val_check_interval=500)

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

        if os.path.exists(args.model_state):
            model.load_state_dict(torch.load(args.model_state))

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
        else:
            trainer.fit(model,
                        DataLoader(train_ds,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   shuffle=True,
                                   collate_fn=dl.TrainDataset.collate_fn))

        # Saving model
        torch.save(model.state_dict(), args.model_state)

    # Load model
    model.load_state_dict(torch.load(args.model_state))
    model.eval()

    _, mse = evaluation(val, args.seq_len, model, output_cols=args.gen_attr,
                        stride=args.stride, context=args.context,
                        pad_both_ends=(not args.no_ctx_train))

    for i, col in enumerate(args.gen_attr):
        print('Validation set MSE for ' + col + ': ' + str(np.mean(mse[:, i])))


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
    parser.add_argument('--vocab-size', type=int, default=92, help='pitch vocabulary size.')
    parser.add_argument('-r', '--lr', type=float, default=1e-5, help='learning rate for training.')
    parser.add_argument('-l', '--seq-len', type=int, default=32, help='number of notes read by model at once.')
    parser.add_argument('-s', '--hidden-size', type=int, default=64, help='size of hidden model layers.')
    parser.add_argument('--dec-layers', type=int, default=1, help='number of recurrent layers in decoder.')
    parser.add_argument('--enc-layers', type=int, default=1, help='number of recurrent layers in encoder.')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='model dropout rate.')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='mini-batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='max. number of training epochs.')
    parser.add_argument('--max-steps', type=int, default=None, help='max. number of training steps.')
    parser.add_argument('--scheduler-step', type=int, default=10000, help='steps between lr decays.')
    parser.add_argument('--lr-decay-by', type=float, default=0.9, help='lr decay rate on scheduler steps.')
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
        val = None
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
                lr_decay_by=args.lr_decay_by,
                dec_layers=args.dec_layers,
                enc_layers=args.enc_layers)

    # Training model
    run_with_args(model, train, val, args)
