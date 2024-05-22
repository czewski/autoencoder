import torch
import torch.nn as nn

def get_mse(input, reconstructed): #reconstruct
    criterion = nn.MSELoss(reduction='mean')
    mse_per_sample = criterion(reconstructed, input)
    return mse_per_sample

def get_rmse(mse_per_sample):
    rmse = torch.sqrt(torch.mean(mse_per_sample))
    return rmse.item()

def get_mae(input, reconstructed):
    mae = torch.mean(torch.abs(reconstructed - input))
    return mae.item()

def evaluate(input, reconstructed):
    mse_per_sample = get_mse(input, reconstructed)
    rmse = get_rmse(mse_per_sample)
    mae = get_mae(input, reconstructed)
    return mse_per_sample, rmse, mae