import time
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from math import ceil
from datetime import date, timedelta

import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import os

from models import MPNN
from utils import generate_new_features, generate_new_batches, read_meta_datasets, AverageMeter


def train(adj, features, y, optimizer, model):
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(adj, features, y, model):
    #output = model(adj, mob, ident)
    output = model(adj, features)
    loss_test = F.mse_loss(output, y)
    return output, loss_test


def train_model(args, meta_labs, meta_graphs, meta_features, meta_y, meta_train, device, model, optimizer):
    model_eta = '../model_eta.pth.tar'
    # model_theta = '../model_theta.pth.tar'

    norm_grad = 0
    for shift in list(range(0, args.ahead)):
        if 0 in meta_train:
            norm_grad += 63 - args.start_exp - shift
        if 1 in meta_train:
            norm_grad += 47 - args.start_exp - shift
        if 2 in meta_train:
            norm_grad += 48 - args.start_exp - shift

    for shift in list(range(0, args.ahead)):
        for train_idx in meta_train:  # regularly reverse loop from train idx to shift
            # labels = meta_labs[train_idx]
            gs = meta_graphs[train_idx]
            features = meta_features[train_idx]
            y = meta_y[train_idx]
            n_samples = len(gs)
            nfeat = features[0].shape[1]
            # n_nodes = gs[0].shape[0]

            for test_sample in range(args.start_exp, n_samples - shift):
                idx_train = list(range(args.window - 1, test_sample))

                adj_train, features_train, y_train = generate_new_batches(gs, features, y, idx_train, 1, shift,
                                                                          args.batch_size, device, test_sample)

                adj_test, features_test, y_test = generate_new_batches(gs, features, y, [test_sample], 1, shift,
                                                                       args.batch_size, device, -1)

                n_train_batches = ceil(len(idx_train) / args.batch_size)
                n_test_batches = 1

                # -------------------- Meta-Train-Training

                # ---- load eta
                checkpoint = torch.load(model_eta)
                model.load_state_dict(checkpoint['state_dict'])

                model.train()
                train_loss = AverageMeter()

                # ------- Train for one epoch
                for batch in range(n_train_batches):
                    output, loss = train(adj_train[batch], features_train[batch], y_train[batch], optimizer, model)
                    train_loss.update(loss.data.item(), output.size(0))

                # ----------- Backprop eta using the test sample
                output, loss = train(adj_test[0], features_test[0], y_test[0], optimizer, model)
                print(
                    "meta train set " + str(train_idx) + " test sample " + str(test_sample) + " theta generalization=",
                    '%03d' % loss.cpu().detach().numpy())

                # ------------ Take delta from the meta training
                w1 = model.conv1.weight.grad.clone()
                b1 = model.bn1.weight.grad.clone()

                w2 = model.conv2.weight.grad.clone()
                b2 = model.bn2.weight.grad.clone()

                f1 = model.fc1.weight.grad.clone()
                f2 = model.fc2.weight.grad.clone()

                model.eval()

                # ----------- Update eta (one gradient per test sample)
                checkpoint = torch.load(model_eta)
                model.load_state_dict(checkpoint['state_dict'])
                model.conv1.weight.data -= args.meta_lr * w1 / norm_grad
                model.bn1.weight.data -= args.meta_lr * b1 / norm_grad
                model.conv2.weight.data -= args.meta_lr * w2 / norm_grad
                model.bn2.weight.data -= args.meta_lr * b2 / norm_grad
                model.fc1.weight.data -= args.meta_lr * f1 / norm_grad
                model.fc2.weight.data -= args.meta_lr * f2 / norm_grad

                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_eta)


def test_model(args, meta_labs, meta_graphs, meta_features, meta_y, meta_test, device, model, optimizer, fw):
    model_eta = '../model_eta.pth.tar'
    model_theta = '../model_theta.pth.tar'
    # -------------------------------------- Meta Test using pre-trained eta
    labels = meta_labs[meta_test]
    gs = meta_graphs[meta_test]
    features = meta_features[meta_test]
    y = meta_y[meta_test]
    nfeat = features[0].shape[1]
    # real_error = []
    n_samples = len(gs)
    result = []

    # pred_tables = {}
    for shift in list(range(0, args.ahead)):
        for test_sample in range(args.start_exp, n_samples - shift):  #

            idx_train = list(range(args.window - 1, test_sample - args.sep))
            idx_val = list(range(test_sample - args.sep, test_sample, 2))
            idx_train = idx_train + list(range(test_sample - args.sep + 1, test_sample, 2))

            adj_train, features_train, y_train = generate_new_batches(gs, features, y, idx_train, 1, shift,
                                                                      args.batch_size, device, test_sample)
            adj_val, features_val, y_val = generate_new_batches(gs, features, y, idx_val, 1, shift, args.batch_size,
                                                                device, test_sample)
            adj_test, features_test, y_test = generate_new_batches(gs, features, y, [test_sample], 1, shift,
                                                                   args.batch_size, device, -1)

            n_train_batches = ceil(len(idx_train) / args.batch_size)
            n_val_batches = 1
            n_test_batches = 1

            stop = False  #
            stuck = False
            while (not stop):  #
                model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                # -------------------- Training
                best_val_acc = 1e9
                val_among_epochs = []

                # ----------------- Load the meta-trained model
                if (not stuck):
                    checkpoint = torch.load(model_eta)
                    model.load_state_dict(checkpoint['state_dict'])
                    # model.fc2 = nn.Linear(args.hidden, 1).to(device) # output layer still trained from scratch
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    # num_ftrs = model.fc2.in_features

                for epoch in range(args.epochs):
                    start = time.time()

                    model.train()
                    train_loss = AverageMeter()

                    # Train for one epoch
                    for batch in range(n_train_batches):
                        output, loss = train(adj_train[batch], features_train[batch], y_train[batch], optimizer, model)
                        train_loss.update(loss.data.item(), output.size(0))

                    # Evaluate on validation set
                    model.eval()

                    output, val_loss = test(adj_val[0], features_val[0], y_val[0], model)
                    val_loss = int(val_loss.detach().cpu().numpy())

                    if (epoch % 50 == 0):
                        print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
                              "val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                    val_among_epochs.append(val_loss)

                    # --------- Stop if stuck
                    if (epoch < 30 and epoch > 10):
                        if (len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1):
                            stuck = True
                            stop = False
                            break

                    if (epoch > args.early_stop):
                        if (len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):  #
                            print("break")
                            stop = True
                            break
                    stop = True

                    # --------- Remember best accuracy and save checkpoint
                    if val_loss < best_val_acc:
                        best_val_acc = val_loss
                        torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, model_theta)  # ------------------ store to meta_eta to share between test_samples

                    scheduler.step(val_loss)

            # ---------------- Testing
            test_loss = AverageMeter()

            checkpoint = torch.load(model_theta)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.eval()

            output, loss = test(adj_test[0], features_test[0], y_test[0], model)
            o = output.cpu().detach().numpy()
            l = y_test[0].cpu().numpy()

            # pred_tables[shift][:,test_sample+shift] = o

            # ------ Store to map plot
            error = np.sum(abs(o - l))
            df = pd.DataFrame({'n': labels.index, 'o': o, 'l': l})
            if not os.path.exists('output'):
                os.makedirs('output')
            df.to_csv("output/out_" + args.country + "_" + str(test_sample) + "_" + str(shift) + ".csv", index=False)

            n_nodes = adj_test[0].shape[0]
            # print(error/n_nodes)
            print(str(test_sample) + " " + args.country + " eta generalization=", "{:.5f}".format(error))
            result.append(error)

        print("{:.5f}".format(np.mean(result)) + ",{:.5f}".format(np.std(result)) + ",{:.5f}".format(
            np.sum(labels.iloc[:, args.start_exp:test_sample].mean(1))))

        fw.write(str(shift) + ",{:.5f}".format(np.mean(result)) + ",{:.5f}".format(np.std(result)) + "\n")

