#Torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#Utils
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv   
import os
import time
import argparse
from tqdm import tqdm

#Local
from utils import utils, dataset, probability_metrics
from models import mlp_narm, lstm_narm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=40, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
args = parser.parse_args()
print(args)

MODEL_VARIATION = "MLP_NARM_"
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    torch.manual_seed(42)

    print('Loading data...')
    train, valid, test = dataset.load_data_narm(args.dataset_path, valid_portion=args.valid_portion, maxlen=19)
    
    train_data = dataset.RecSysDatasetNarm(train)
    valid_data = dataset.RecSysDatasetNarm(valid)
    test_data = dataset.RecSysDatasetNarm(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = utils.collate_fn_narm)
    #valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn_narm)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn_narm)

    if args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 43098
    elif args.dataset_path.split('/')[-2] in ['yoochoose1_64', 'yoochoose1_4']:
        n_items = 37484
    else:
        raise Exception('Unknown Dataset!')


    #model = mlp_narm.MLP(n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device) 
    model = lstm_narm.LSTM(n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device) 

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    # Info
    losses = []
    now = datetime.now()
    now_time = time.time()
    timestamp = now.strftime("%d_%m_%Y_%H:%M:%S")

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        sum_epoch_loss = trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)

        # Calculate the average loss for the epoch
        epoch_loss = sum_epoch_loss/len(train_loader)
        losses.append(epoch_loss)
        
        #recall, mrr = validate(valid_loader, model)
        #print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'checkpoints/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')

    # Loss curve
    print('--------------------------------')
    print('Plotting loss curve...')
    plt.clf()
    plt.plot(losses[1:])
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')  
    plt.savefig('loss_curves/'+MODEL_VARIATION+'loss_curve_'+timestamp+'.png')

    # Test model
    ckpt = torch.load('checkpoints/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    test_recall, test_mrr, test_hit = validate(test_loader, model)
    print("Test: Recall@20: {:.4f}, MRR@20: {:.4f}, HIT@20: {:.4f}".format(test_recall, test_mrr, test_hit))

    # Save test metrics to stats
    model_unique_id = MODEL_VARIATION + timestamp
    fields=[model_unique_id, test_recall, test_mrr, test_hit,timestamp,(time.time() - now_time), test_recall, test_mrr, test_hit]  
    with open(r'stats/data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()
    
    return sum_epoch_loss


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    hits = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)

            logits = F.softmax(outputs, dim = 1)
            recall, mrr, hit = probability_metrics.evaluate(logits, target, k = args.topk)

            recalls.append(recall)
            mrrs.append(mrr)
            hits.append(hit)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_hit = np.mean(hit)
    return mean_recall, mean_mrr, mean_hit


if __name__ == '__main__':
    main()