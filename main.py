import typing
from io import StringIO
from typing import Tuple
import json
import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.utils.data as Data

import utils
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")
root_path = "/home/rr/Downloads/nsm_data/Train/"
total = 1753  #1753
files_num = 50


def da_rnn(encoder_hidden_size=64, decoder_hidden_size=64, T=10, learning_rate=0.01, batch_size=128):
    train_cfg = TrainConfig(T, int(files_num * 100 * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {int(total / files_num) * train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": 5307, "hidden_size": encoder_hidden_size, "T": T}
    # enc_kwargs = {"input_size": 5307, "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": 618}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.001)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.001)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(inputs_list, net: DaRnnNet, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    f = open(root_path+"test.txt", "w")
    iter_per_epoch = int(np.ceil(int(total / files_num) * t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros((n_epochs+1) * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs+1)
    logger.info(
        f"Iterations per epoch: {int(total / files_num) * t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0
    scale = StandardScaler()
    for e_i in range(n_epochs):

        train_input_data = pd.DataFrame()
        train_label_data = pd.DataFrame()
        all_val_loss = list()
        all_test_loss = list()
        n_iter_per_epoche = 0

        for i, file in enumerate(tqdm(inputs_list)):
            if i % files_num:  # 50
                single_input_data = pd.read_csv(root_path + "Input/" + file, sep=' ', header=None, dtype=float)
                single_label_data = pd.read_csv(root_path + "Label/" + file, sep=' ', header=None, dtype=float)
                train_input_data = train_input_data.append(single_input_data, ignore_index=True)
                train_label_data = train_label_data.append(single_label_data, ignore_index=True)
            elif i != 0 and i % files_num == 0:

                scale = scale.fit(train_input_data)
                t_input_data = scale.transform(train_input_data)
                # t_input_data = np.array(train_input_data)
                t_label_data = np.array(train_label_data)

                perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
                for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
                    batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
                    feats, y_history, y_target = prep_train_data(batch_idx, t_cfg,
                                                                 t_input_data[:t_cfg.train_size],
                                                                 t_label_data[:t_cfg.train_size])

                    loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
                    iter_losses[e_i * iter_per_epoch + n_iter_per_epoche] = loss
                    # print('itr_loss:', loss)

                    n_iter += 1
                    n_iter_per_epoche += 1
                    adjust_learning_rate(net, n_iter)

                y_val_pred = predict(net, t_input_data[:t_cfg.train_size], t_label_data[:t_cfg.train_size],
                                     t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                     on_train=True)
                y_test_pred = predict(net, t_input_data, t_label_data,
                                      t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                      on_train=False)
                # TODO: make this MSE and make it work for multiple inputs
                val_loss = [x - y for x, y in zip(y_val_pred, t_label_data[t_cfg.T-1:]) if x.all() != 0]
                test_loss = [x - y for x, y in zip(y_test_pred, t_label_data[t_cfg.train_size:]) if x.all() != 0]
                all_val_loss = all_val_loss + val_loss
                all_test_loss = all_test_loss + test_loss

                train_input_data = train_input_data.drop(train_input_data.index, inplace=False)
                train_label_data = train_label_data.drop(train_label_data.index, inplace=False)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 1 == 0:
            f.write(str([e_i, epoch_losses[e_i], np.mean(np.abs(all_val_loss)), np.mean(np.abs(all_test_loss))]))
            f.flush()
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.5f}, val loss: {np.mean(np.abs(all_val_loss))}."
                        f"test loss: {np.mean(np.abs(all_test_loss))}.")
        if e_i % 20 == 0 and e_i != 0:
            torch.save(net.encoder.state_dict(), os.path.join("data", "encoder"+str(e_i)+".torch"))
            torch.save(net.decoder.state_dict(), os.path.join("data", "decoder"+str(e_i)+".torch"))

    f.close()
    return iter_losses, epoch_losses


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, input_data, label_data):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, input_data.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, label_data.shape[1]))
    y_target = label_data[batch_idx + t_cfg.T]
    # y_target = label_data.iloc[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = input_data[b_slc, :]
        y_history[b_i, :] = label_data[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)

    # regularization_loss = 0
    # for param in t_net.decoder.parameters():
    #     regularization_loss += torch.sum(abs(param))
    #
    # classify_loss = loss_func(y_pred, y_true)
    # loss = classify_loss + 0.5 * regularization_loss

    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(t_net: DaRnnNet, input_data, label_data, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = label_data.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((input_data.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, input_data.shape[1]))
        y_history = np.zeros((b_len, T - 1, label_data.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = input_data[idx, :]
            y_history[b_i, :] = label_data[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


def main():
    save_plots = True

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    da_rnn_kwargs = {"batch_size": 64, "T": 10}
    config, model = da_rnn(learning_rate=.001, **da_rnn_kwargs)
    iter_loss, epoch_loss = train(inputs_list, model, config, n_epochs=100, save_plots=save_plots)
    # final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

    plt.figure()
    plt.semilogy(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    # plt.figure()
    # plt.plot(final_y_pred, label='Predicted')
    # plt.plot(data.targs[config.train_size:], label="True")
    # plt.legend(loc='upper left')
    # utils.save_or_show_plot("final_predicted.png", save_plots)

    with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
        json.dump(da_rnn_kwargs, fi, indent=4)

    # joblib.dump(scaler, os.path.join("data", "scaler.pkl"))

    torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
    torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))


if __name__ == '__main__':
    main()
