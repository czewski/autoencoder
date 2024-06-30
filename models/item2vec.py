from gensim.models import Word2Vec
import torch
import pandas as pd
from ast import literal_eval
import numpy as np

# Save/Train (?)
train_set = pd.read_csv('data/diginetica_my_preprocess_target_padding/full_padding.csv', sep=',')
train_set['padded_itemId'] = train_set['padded_itemId'].apply(lambda s: literal_eval(s))

embedding_dim = 100
#43098
# 107312

# Train the item2vec model
item2vec_model = Word2Vec(train_set['padded_itemId'], vector_size=embedding_dim, window=5, min_count=1, sg=1)

# Save the model if needed
item2vec_model.save("item2vec100.model")


## Load
# # Load the trained model
# item2vec_model = Word2Vec.load("item2vec.model")

# # Create a dictionary of item embeddings
# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}

# # Convert the dictionary of embeddings to a NumPy array
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])

# # Convert the NumPy array to a PyTorch tensor
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
