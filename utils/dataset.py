import pickle
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from ast import literal_eval
import torch


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


def load_data_target(root, maxlen=15, sort_by_len=False):

    # Load the dataset
    path_train_data = root + 'train.txt'
    path_test_data = root + 'test.txt'
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y


    (train_set_x, train_set_y) = train_set
    (test_set_x, test_set_y) = test_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    test = (test_set_x, test_set_y)

    return train, test

def load_data_mlp(root):
    path_train_data = root + 'train_padding_reorder.csv'
    path_test_data = root + 'test_padding_reorder.csv'

    train_set = pd.read_csv(path_train_data, sep=',')
    train_set['padded_itemId'] = train_set['padded_itemId'].apply(lambda s: literal_eval(s))

    test_set = pd.read_csv(path_test_data, sep=',')
    test_set['padded_itemId'] = test_set['padded_itemId'].apply(lambda s: literal_eval(s))

    return train_set, test_set

class DatasetMLP(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.itemIds = self.data['padded_itemId']
        self.targets = self.data['target']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        itemId = torch.tensor(self.itemIds[idx])
        target = torch.tensor(self.targets[idx])
        return itemId, target

# This is used for session only (reconstruct)
class DigineticaReconstruct(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        session_items = self.data[index]
        #target_item = self.data[1][index]
        return session_items

    def __len__(self):
        return len(self.data) #[0]
    
    # This is used for session only (reconstruct)
class DigineticaTarget(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0]) 