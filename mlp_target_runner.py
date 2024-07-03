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
from gensim.models import Word2Vec

# Local
from models import mlp
from utils import dataset, target_metric, utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica_my_preprocess_target_padding_reorder/')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
#parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()

MODEL_VARIATION = "MLP_TARGET_"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(42)

    ## Data loading and create dataloaders
    train, test = dataset.load_data_mlp(args.dataset_path) 

    train_data = dataset.DatasetMLP(train)
    test_data = dataset.DatasetMLP(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True) # , collate_fn=utils.collate_fn
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = True) 

    ## Load Embedding Matrix
    item2vec_model = Word2Vec.load("item2vec100_reorder.model")
    item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}
    embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

    ## Model definitions
    model = mlp.MLP(embedding_matrix, input_dim=5, hidden_dim=500).to(device) 
    optimizer = optim.Adam(model.parameters(), args.lr) 
    criterion = nn.CrossEntropyLoss() #nn.MSELoss() ##
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    # Info
    losses = []
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H:%M:%S")

    # Training loop
    for epoch in tqdm(range(args.epoch)):
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        log_aggr = 100

        for i, (seq, target)  in tqdm(enumerate(train_loader)):
            seq = seq.to(device)
            target = target.to(device).to(torch.float32)
            optimizer.zero_grad()

            outputs = model(seq)
            loss = criterion(outputs, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
 
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

        # recall, mrr, hit = validate(test_loader, model)
        # print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, HIT@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, args.topk, hit))

        # store best loss and save a model checkpoint
        # ckpt_dict = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }

        # torch.save(ckpt_dict, 'checkpoints/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')

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
    test_mse, test_rmse, test_mae = validate(test_loader, model)
    print("Test: MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(test_mse, test_rmse, test_mae))

    # Save stats to csv
    model_unique_id = MODEL_VARIATION + timestamp
    fields=[model_unique_id, test_mse, test_rmse, test_mae,timestamp,(time.time() - start), test_mse, test_rmse, test_mae]  
    with open(r'stats/data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def validate(valid_loader, model): #Can be used either for test or valid
    model.eval()
    mses = []
    rmses = []
    maes = []

    with torch.no_grad():
        for _, (seq, target, lens)  in tqdm(enumerate(valid_loader)):
            seq = seq.to(device).to(torch.float32)
            outputs = model(seq)
            mse, rmse, mae = reconstruct_metric.evaluate(outputs, target)
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