# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:54
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from model import *
import timeit


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    import os
    from model import Encoder, Decoder, DecoderLayer, SelfAttention, PositionwiseFeedforward, Predictor, Tester

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # DATASET = "GPCR_train"

    model = Word2Vec.load("transformerCPI_SICBC/GPCR/word2vec_30.model")
    seq = 'MYNGSCCRIEGDTISQVMPPLLIVAFVLGALGNGVALCGFCFHMKTWKPSTVYLFNLAVADFLLMICLPFRTDYYLRRRHWAFGDIPCRVGLFTLAMNRAGSIVFLTVVAADRYFKVVHPHHAVNTISTRVAAGIVCTLWALVILGTVYLLLENHLCVQETAVSCESFIMESANGWHDIMFQLEFFMPLGIILFCSFKIVWSLRRRQQLARQARMKKATRFIMVVAIVFITCYLPSVSARLYFLWTVPSSACDPSVHGALHITLSFTYMNSMLDPLVYYFSSPSFPKFYNKLKICSLKPKQPGHSKTQRPEEMPISNLGRRSCISVANSFQSQSDGQWDPHIVEWH'
    protein_embedding = get_protein_embedding(model, seq_to_kmers(seq))
    protein = torch.FloatTensor(protein_embedding).unsqueeze(0).to(device)

    for dataset_idx in range(1, 24):
        DATASET = 'split_' + str(dataset_idx).zfill(3) + '/'
        print('Dataset:', dataset_idx)

        """CPU or GPU"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('The code uses GPU...')
        else:
            device = torch.device('cpu')
            print('The code uses CPU!!!')

        for idx in range (1, 13):
            """Load preprocessed data."""
            print('index:', idx)
            dir_input = ('data/dataset/' + DATASET)
            print('Loading compunds...')
            compounds_path = dir_input + 'compounds_part' + str(idx) + '.npy'
            if not os.path.exists(compounds_path):
                break

            compounds = load_tensor(dir_input + 'compounds_part' + str(idx), torch.FloatTensor)
            # print(len(compounds))
            print('Loading adjacencies...')
            adjacencies = load_tensor(dir_input + 'adjacencies_part' + str(idx), torch.FloatTensor)
            # proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
            # interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

            # interactions = [torch.tensor(0.0) for _ in range(compounds.__len__())]

            print('Loading proteins')

            """Create a dataset and split it into train/dev/test."""
            # dataset = list(zip(compounds, adjacencies, proteins, interactions))
            # dataset = shuffle_dataset(dataset, 1234)
            # dataset_train, dataset_dev = split_dataset(dataset, 0.8)

            dataset_test = list(zip(compounds, adjacencies))
            print('length of dataset_test:', len(dataset_test))

            print('Finished zipping data...')

            """ create model ,trainer and tester """
            protein_dim = 100
            atom_dim = 34
            hid_dim = 64
            n_layers = 3
            n_heads = 8
            pf_dim = 256
            dropout = 0.1
            batch = 64
            lr = 1e-4
            weight_decay = 1e-4
            decay_interval = 5
            lr_decay = 1.0
            iteration = 300
            kernel_size = 7

            encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
            decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
            model = Predictor(encoder, decoder, device)
            model.load_state_dict(torch.load("transformerCPI_SICBC/model/GPCR.pt"))
            model.to(device)
            # print(model)
            # print(dataset_test[0])
            # print(len(dataset_test))
            # trainer = Trainer(model, lr, weight_decay, batch)
            
            tester = Tester(model)
            tester.test(dataset_test, dataset_idx, idx)
            print('Finished testing...')
            

