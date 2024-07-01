from gensim.models import Word2Vec
import torch
import pandas as pd
from ast import literal_eval
import numpy as np

import pandas as pd
from collections import Counter

# Save/Train (?)
train_set = pd.read_csv('data/diginetica_my_preprocess_target_padding_reorder/full_padding_reorder.csv', sep=',')
train_set['padded_itemId'] = train_set['padded_itemId'].apply(lambda s: literal_eval(s))

# Flatten the lists into a single list
all_items = [item for sublist in train_set['padded_itemId'] for item in sublist]

# Find the number of unique items
unique_items = set(all_items)
num_unique_items = len(unique_items)

print(f'Number of unique items: {num_unique_items}')

print(train_set['padded_itemId'])
print('unique in dataset: 120778')

print('sessions: ', len(train_set['padded_itemId']))

embedding_dim = 100
#43098
# 107312

# Train the item2vec model
item2vec_model = Word2Vec(sentences=train_set['padded_itemId'], 
                          vector_size=embedding_dim, 
                          window=5, 
                          min_count=0,  #appear more than 1 times to be keeped
                          sg=1, #Skip-gram
                          epochs=5,
                          max_vocab_size=None,
                          seed=42)

# Save the model if needed
item2vec_model.save("item2vec100_reorder_full.model")

print('unique itens: ', len(item2vec_model.wv))

## Load
# # Load the trained model
# item2vec_model = Word2Vec.load("item2vec.model")

# # Create a dictionary of item embeddings
# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}

# # Convert the dictionary of embeddings to a NumPy array
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])

# # Convert the NumPy array to a PyTorch tensor
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
