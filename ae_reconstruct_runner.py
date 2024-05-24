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

# Local
from models import autoencoder
from utils import dataset, target_metric, utils, reconstruct_metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica_normalized_padded_no_target/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  
#parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
#parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
#parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(42)
    train, test = dataset.load_data(args.dataset_path) 

    train_data = dataset.DigineticaReconstruct(train)
    test_data = dataset.DigineticaReconstruct(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True) #, collate_fn = utils.collate_fn
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False) #, collate_fn = utils.collate_fn

    model = autoencoder.AutoEncoder().to(device) #    n_items = 43098
    optimizer = optim.Adam(model.parameters(), args.lr) #optim.RMSprop
    criterion = nn.MSELoss() #nn.KLDivLoss() nn.CrossEntropyLoss() nn.BCELoss()

    losses = []
    for epoch in tqdm(range(args.epoch)):
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        log_aggr = 100

        for i, seq  in tqdm(enumerate(train_loader)):
            print(seq.size())
            seq = seq.to(device).to(torch.float32)
            optimizer.zero_grad()
            outputs = model(seq)

            #Create mask max=20, real=lens[x]
            #mask = utils.create_matrix_mask(len(seq), 20, lens)
            #outputs = outputs * mask

            loss = criterion(outputs, seq)
            loss.backward()
            optimizer.step() 

            loss_val = loss.item()
            sum_epoch_loss += loss_val

            # Calculate the current iteration number
            #iter_num = epoch * len(train_loader) + i + 1
        
            if i % log_aggr == 0:
                print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'% (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1),len(seq) / (time.time() - start)))

            start = time.time()

        # Calculate the average loss for the epoch
        epoch_loss = sum_epoch_loss/len(train_loader)
        losses.append(epoch_loss)

        # recall, mrr, hit = validate(test_loader, model)
        # print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, HIT@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, args.topk, hit))

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

    # Evaluate 
    print(validate(test_loader, model=model))

def validate(valid_loader, model): #Can be used either for test or valid
    model.eval()
    mses = []
    rmses = []
    maes = []

    with torch.no_grad():
        for _, seq  in tqdm(enumerate(valid_loader)):
            seq = seq.to(device).to(torch.float32)
            outputs = model(seq)
            mse, rmse, mae = reconstruct_metric.evaluate(outputs, seq)
            mses.append(mse.item())
            rmses.append(rmse)
            maes.append(mae)

            #print(mse) 

    mean_mse = np.mean(mses)
    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)

    return mean_mse, mean_rmse, mean_mae

if __name__ == '__main__':
    main()