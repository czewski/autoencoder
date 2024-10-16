from gensim.models import Word2Vec
import torch
import pandas as pd
#from ast import literal_eval
import numpy as np
import pickle
from utils import dataset

# diginetica -> 43099
# yoochoose -> 37485

## Train word2vec model
root = "data/diginetica/" #yoochoose1_4 #diginetica
train, valid, test = dataset.load_data_narm(root, valid_portion=0.1, maxlen=15)
embedding_dim = 50
item2vec_model = Word2Vec(train[0], vector_size=embedding_dim, window=10, 
                          min_count=1, sg=1, seed=522, epochs=50)
item2vec_model.save("embeddings/diginetica/item2vec.model")


# Add missing ids to word2vec model
model = Word2Vec.load("embeddings/diginetica/item2vec.model")
total_ids = set(range(43099))
trained_ids = set(int(key) for key in model.wv.key_to_index.keys())
missing_ids = total_ids - trained_ids

# print(missing_ids)

# vector_size = model.vector_size
# for missing_id in missing_ids:
#     random_vector = np.random.randn(vector_size)
#     model.wv.add_vector(missing_id, random_vector)

# Save the updated model
model.save("embeddings/diginetica/item2vec.model")


## Create weight matrix
# # Load the trained model
# item2vec_model = Word2Vec.load("item2vec.model")

# # Create a dictionary of item embeddings
# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}

# # Convert the dictionary of embeddings to a NumPy array
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])

# # Convert the NumPy array to a PyTorch tensor
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)