#py utils
import os
import time
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join

#torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

#sklearn
from sklearn.model_selection import KFold

#local dep
import metric
from utils import collate_fn
from dataset import load_data, RecSysDataset
from autoencoder import Encoder
#from testautoencoder import Encoder

#log metrics
import matplotlib.pyplot as plt
import time
import csv   
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'AUTOENCODER'

def main():
    # Set fixed random number seed
    torch.manual_seed(42)

    k_folds = 5    
    kfold = KFold(n_splits=k_folds, shuffle=False)

    print('Loading data...')
    train, test = load_data(args.dataset_path) #valid_portion=args.valid_portion

    test_data = RecSysDataset(test)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    #fix kfold
    X_train, Y_train = train  # Assuming train is in the format (X, Y) 

    n_items = 43098
    model = Encoder(L=[512, 19, 10]).to(device) #n_items = n_items, embedding_dim=50, hidden_size=100

    # Create valid from fold 
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(X_train)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Use train_idx to index into both X_train and Y_train for training data
        train_X_fold, train_Y_fold = [X_train[i] for i in train_ids], [Y_train[i] for i in train_ids]
        x = train_X_fold, train_Y_fold
        train_data_fold = RecSysDataset(x)
        train_loader_fold = DataLoader(train_data_fold, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)

        # Use valid_idx to index into both X_train and Y_train for validation data
        valid_X_fold, valid_Y_fold = [X_train[i] for i in valid_ids], [Y_train[i] for i in valid_ids]
        y = valid_X_fold, valid_Y_fold
        valid_data_fold = RecSysDataset(y) 
        valid_loader_fold = DataLoader(valid_data_fold, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

        optimizer = optim.Adam(model.parameters(), args.lr)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss()
        scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

        start_time = time.time()
        losses = []
        for epoch in tqdm(range(args.epoch)):
            # train for one epoch
            epoch_loss, _ = trainForEpoch(train_loader_fold, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)
            scheduler.step(epoch = epoch)
            
            losses.append(epoch_loss)

            # recall, mrr, hit = validate(valid_loader_fold, model)
            # print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, HIT@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, args.topk, hit))

            #print('EPOCH GRAD')
            #print(epoch_gradients)

            # store best loss and save a model checkpoint
            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(ckpt_dict, model_name+'_latest_checkpoint'+str(fold)+'.pth.tar')
        

        print('--------------------------------')
        print('Train {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr))

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
     # Set the model to training mode
    model.train()
    sum_epoch_loss = 0
    start = time.time()

    # List to store gradients for each epoch
    epoch_gradients = []

    # Iterate over batches in the training data loader
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        print("seq")
        print(seq)

        print("seq size")
        print(seq.size())
        # Move data to the device (e.g., GPU)
        seq = seq.to(device)
        target = target.to(device)

        # Zero the gradients in the optimizer
        optimizer.zero_grad()

        # Forward pass: compute model predictions
        outputs = model(seq, lens)

        print("target.size()")
        print(target.size())

        print("outputs size")
        print(outputs.size())

        # print("Target sample 5:", target[5])
        # time.sleep(2)

        # print("Output sample 5:", outputs[5])
        # time.sleep(2)

        # Calculate the loss between predictions and actual targets
        #loss = criterion(outputs, target)
        loss = criterion(outputs, target.float())

        # Backward pass: compute gradients and update model parameters
        loss.backward()
        optimizer.step() 

        # Get the numerical value of the loss function
        loss_val = loss.item()
        sum_epoch_loss += loss_val

        # Calculate the current iteration number
        iter_num = epoch * len(train_loader) + i + 1

        # Print training information at specified intervals
        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()

    # Calculate the average loss for the epoch
    epoch_loss = sum_epoch_loss/len(train_loader)
    return epoch_loss, epoch_gradients


if __name__ == '__main__':
    main()
