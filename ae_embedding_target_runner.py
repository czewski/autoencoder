# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

#Utils
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv   

# Local
from models import autoencoder_embedding
from utils import dataset, target_metric, utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica_with_target/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_VARIATION = "AE_EMBEDDING_TARGET_"

def main():
    torch.manual_seed(42)
    train, test = dataset.load_data_target(args.dataset_path) 

    train_data = dataset.DigineticaTarget(train)
    test_data = dataset.DigineticaTarget(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True,collate_fn = utils.collate_fn) #, collate_fn = utils.collate_fn
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn) #, 

    model = autoencoder_embedding.AutoEncoderEmbedding(device).to(device) #   
    optimizer = optim.Adam(model.parameters(), args.lr) #optim.RMSprop
    criterion = nn.CrossEntropyLoss() #nn.KLDivLoss() nn.CrossEntropyLoss() nn.BCELoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    losses = []
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H:%M:%S")
    for epoch in tqdm(range(args.epoch)):
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        log_aggr = 100
        #print(train_loader)
        for i, (seq, target, lens)  in tqdm(enumerate(train_loader)):
            seq = seq.to(device)
            target = target.to(device)

            # print("seq")
            # print(seq.size())
            # print("target")
            # print(target.size())

            optimizer.zero_grad()
            # print(seq[0])
            # print(lens[0])

            outputs = model(seq, lens)
            # print(outputs.size())
            # print(target.size())

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step() 
            scheduler.step()


            loss_val = loss.item()
            sum_epoch_loss += loss_val
        
            if i % log_aggr == 0:
                print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'% (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1),len(seq) / (time.time() - start)))

            start = time.time()

        # Calculate the average loss for the epoch
        epoch_loss = sum_epoch_loss/len(train_loader)
        losses.append(epoch_loss)

        recall, mrr, hit = validate(test_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, HIT@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, args.topk, hit))

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
    print("Test: RECALL@20: {:.4f}, MRR@20: {:.4f}, HIT@20: {:.4f}".format(test_recall, test_mrr, test_hit))

    # Save stats to csv
    model_unique_id = MODEL_VARIATION + timestamp
    fields=[model_unique_id, test_recall, test_mrr, test_hit, timestamp,(time.time() - start), test_recall, test_mrr, test_hit]  
    with open(r'stats/data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def validate(valid_loader, model): #Can be used either for test or valid
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

            recall, mrr, hit = target_metric.evaluate(logits, target, args.topk)

            if hit != 0:
                print(hit)

            recalls.append(recall)
            mrrs.append(mrr)
            hits.append(hit)

    mean_recall = np.mean(recalls)
    mean_mrrs = np.mean(mrrs)
    mean_hit = np.mean(hits)

    return mean_recall, mean_mrrs, mean_hit

if __name__ == '__main__':
    main()