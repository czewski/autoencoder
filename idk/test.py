import torch
import pickle 
from gensim.models import Word2Vec
import numpy as np


# Assuming you have a Word2Vec model already trained
model = Word2Vec.load("embeddings/item2vec_05_08.model")

# Total IDs
total_ids = set(range(43099))

# IDs in the model
trained_ids = set(int(key) for key in model.wv.key_to_index.keys())

# Find missing IDs
missing_ids = total_ids - trained_ids

print(missing_ids)

vector_size = model.vector_size
for missing_id in missing_ids:
    random_vector = np.random.randn(vector_size)
    model.wv.add_vector(missing_id, random_vector)

# Save the updated model
model.save("embeddings/item2vec_06_08.model")

# ## Load Embedding Matrix
# item2vec_model = Word2Vec.load("embeddings/item2vec_05_08.model")
# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
# #embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

# print(max(item2vec_model.wv.index_to_key))
# print(len(item_embeddings))
# print(item2vec_model.wv)

43098

# root = "data/diginetica/"
# path_train_data = root + 'train.txt'
# path_test_data = root + 'test.txt'
# with open(path_train_data, 'rb') as f1:
#     train_set = pickle.load(f1)

# with open(path_test_data, 'rb') as f2:
#     test_set = pickle.load(f2)

# print(len(train_set))
# print(len(train_set[0]))
# print(len(train_set[1]))


# Test model
# ckpt = torch.load('checkpoints/LSTM_ATT_latest_checkpoint_02_08_2024_14:30:46.pth.tar')
# print(ckpt['epoch'])
