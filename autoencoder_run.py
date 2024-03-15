# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Utils
import argparse
from tqdm import tqdm
import time
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# import pandas as pd

# Local
from models import autoencoder
from utils import dataset, utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Set fixed random number seed
    torch.manual_seed(42)

    # Load data
    train, test = dataset.load_data(args.dataset_path) 
    xxxx= len(train)
    yyyy= len(test)

    # Init dataloaders
    train_data = dataset.RecSysDataset(train)
    test_data = dataset.RecSysDataset(test)

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = utils.collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn)

    # Init models
    n_items = 43098
    model = autoencoder.AutoEncoder(n_items).to(device) 

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()     #criterion = nn.MSELoss()

    for epoch in tqdm(range(args.epoch)):
        model.train()
        sum_epoch_loss = 0
        start = time.time()

        # Iterate over batches in the training data loader
        for i, (seq, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #Why is it a tuple and not a fucking tensor?
            seq = seq.to(device)

            optimizer.zero_grad()
            outputs = model(seq)

            # aqui preciso fazer entre input (seq) e output
            loss = criterion(outputs, seq)
            loss.backward()
            optimizer.step() 

            # Get the numerical value of the loss function
            loss_val = loss.item()
            sum_epoch_loss += loss_val

            # Calculate the current iteration number
            iter_num = epoch * len(train_loader) + i + 1

            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1),
                    len(seq) / (time.time() - start)))
        
        start = time.time()

if __name__ == '__main__':
    main()