"""
create on 18 Sep, 2019

@author: wangshuo

Reference: https://github.com/lijingsdu/sessionRec_NARM/blob/master/data_process.py
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

def load_data(root, maxlen=20, sort_by_len=False): #valid_portion=0.1,
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    # Load the dataset
    path_train_data = root + 'train.txt'
    path_test_data = root + 'test.txt'
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)


    if maxlen:
        new_train_set_x = []
        #new_train_set_y = []
        for x in train_set:
            if len(x) < maxlen:
                new_train_set_x.append(x)
                #new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                #new_train_set_y.append(y)
        train_set = (new_train_set_x ) #new_train_set_y
        del new_train_set_x#, new_train_set_y

        new_test_set_x = []
        #new_test_set_y = []
        for xx in test_set:#[0], test_set[1]
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                #new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                #new_test_set_y.append(yy)
        test_set = (new_test_set_x) ##, new_test_set_y
        del new_test_set_x##, new_test_set_y

    train = train_set
    test = test_set

    return train, test

# This is used for session only (reconstruct)
class RecSysDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        session_items = self.data[index]
        #target_item = self.data[1][index]
        return session_items

    def __len__(self):
        return len(self.data) #[0]