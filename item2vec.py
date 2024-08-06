from gensim.models import Word2Vec
import torch
import pandas as pd
#from ast import literal_eval
import numpy as np
import pickle
from utils import dataset

# Save/Train (?)
root = "data/diginetica/"
train, valid, test = dataset.load_data_narm(root, valid_portion=0.1, maxlen=15)

# path_test_data = root + 'test.txt'
# with open(path_test_data, 'rb') as f2:
#     test_set = pickle.load(f2)
#43098
# 107312

embedding_dim = 50

# Train the item2vec model
item2vec_model = Word2Vec(train[0], vector_size=embedding_dim, window=10, 
                          min_count=1, sg=1, seed=522, epochs=25)

# Save the model if needed
item2vec_model.save("embeddings/item2vec_05_08.model")


## Load
# # Load the trained model
# item2vec_model = Word2Vec.load("item2vec.model")

# # Create a dictionary of item embeddings
# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}

# # Convert the dictionary of embeddings to a NumPy array
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])

# # Convert the NumPy array to a PyTorch tensor
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)