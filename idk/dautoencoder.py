# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#Utils
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# import pandas as pd

# Local
from models import vautoencoder
from utils import dataset, target_metric, utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=5, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  
#parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
#parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
#parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    torch.manual_seed(42)
    train, test = dataset.load_data(args.dataset_path) 

    train_data = dataset.RecSysDataset(train)
    test_data = dataset.RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = utils.collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn)

    n_items = 43098
    model = vautoencoder.VariationalAutoencoder(n_items, input_dim=19, hidden_dim=100).to(device)

    optimizer = optim.Adam(model.parameters(), args.lr)
    #optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.MSELoss() #nn.KLDivLoss() # #nn.CrossEntropyLoss()     
    #criterion = nn.BCELoss()
    noise_factor = 0.5

    losses = []
    for epoch in tqdm(range(args.epoch)):
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        log_aggr = 100
        
        # Iterate over batches in the training data loader
        for i, (seq, lens) in tqdm(enumerate(train_loader)):
            seq = seq.to(device).to(torch.float32)
            
            #Probably will need to adjust dimension here
            inputs_noisy = seq + noise_factor * torch.randn_like(seq)
            inputs_noisy = torch.clamp(inputs_noisy, 0., 1.)  # Ensure pixel values are in [0, 1]
            inputs_noisy = inputs_noisy.view(inputs_noisy.size(0), -1)

            optimizer.zero_grad()
            outputs, mu, log_var = model(seq)
            recon_loss = criterion(seq, outputs)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_divergence
            loss.backward()
            optimizer.step() 

            # Get the numerical value of the loss function
            loss_val = loss.item()
            sum_epoch_loss += loss_val

            # Calculate the current iteration number
            iter_num = epoch * len(train_loader) + i + 1
        
            if i % log_aggr == 0:
                #print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'% (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1),len(seq) / (time.time() - start)))
                print(f'Epoch [{epoch + 1}/{args.epoch}], Recon Loss: {recon_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}')

            start = time.time()


        # Calculate the average loss for the epoch
        epoch_loss = sum_epoch_loss/len(train_loader)
        losses.append(epoch_loss)

        #recall, mrr, hit = validate(valid_loader_fold, model)
        #print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, HIT@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, args.topk, hit))

        # store best loss and save a model checkpoint
        #ckpt_dict = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }

        #torch.save(ckpt_dict, 'autoencoder_latest_checkpoint.pth.tar')

    # Loss curve
    print('--------------------------------')
    print('Plotting loss curve...')
    plt.clf()
    plt.plot(losses[1:])
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')  
    plt.savefig('loss_curve.png')

if __name__ == '__main__':
    main()

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
            recall, mrr, hit = target_metric.evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
            hits.append(hit)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_hit = np.mean(hits)
    return mean_recall, mean_mrr, mean_hit