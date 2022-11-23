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

from gnn_trainers import train_model, test_model


def parser_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=300,
    #                     help='Number of epochs to train.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur', default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    parser.add_argument('--meta-lr', default=0.01,  # 0.001,
                        help='')

    # ---------------- Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    args = parser.parse_args()
    return args


def meta_assign(country):
    if country == "IT":
        return [1, 2, 3], 0
    elif country== "ES":
        return [0, 2, 3], 1
    elif country == "EN":
        return [0, 1, 3], 2
    else:
        return [0, 1, 2], 3


if __name__ == '__main__':
    args = parser_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window)

    train_countries = ["IT", "ES", "EN", "FR"]
    for args.country in train_countries:
        meta_train, meta_test = meta_assign(args.country)

        nfeat = meta_features[0][0].shape[1]

        model_eta = '../model_eta.pth.tar'
        # model_theta = '../model_theta.pth.tar'

        if not os.path.exists('../results'):
            os.makedirs('../results')
        fw = open("../results/results_"+args.country+"_tl_.csv","a")#results/
        fw.write("shift,loss,loss_std\n")

        # ------ Initialize the model
        model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }, model_eta)

        print("Meta Train")
        train_model(args, meta_labs, meta_graphs, meta_features, meta_y, meta_train, device, model, optimizer)
        print("Meta Test")
        test_model(args, meta_labs, meta_graphs, meta_features, meta_y, meta_test, device, model, optimizer, fw)

        fw.close()










