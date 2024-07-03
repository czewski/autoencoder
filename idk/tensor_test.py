# Torch
import torch
import torch.nn as nn

#Utils
import numpy as np
from gensim.models import Word2Vec


## Load Embedding Matrix
item2vec_model = Word2Vec.load("item2vec100_reorder_full.model")

print(len(item2vec_model.wv.index_to_key))


# Create a word index dictionary (mapping each word to a unique index)
word_index = {word: idx for idx, word in enumerate(item2vec_model.wv.index_to_key, start=1)}

# Initialize the embedding matrix with zeros
embedding_dim = item2vec_model.vector_size
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

# Fill the embedding matrix with word vectors from the Word2Vec model
for word, idx in word_index.items():
    if word in item2vec_model.wv:
        embedding_matrix[idx] = item2vec_model.wv[word]


# item_embeddings = {item: item2vec_model.wv[item] for item in item2vec_model.wv.index_to_key}
# embedding_matrix = np.array([item_embeddings[item] for item in sorted(item_embeddings.keys())])
# embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

print(len(word_index))

# Convert the embedding matrix to a PyTorch tensor
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

embedding = nn.Embedding.from_pretrained(embedding_matrix)
# embedding = nn.Embedding(107311, 100, padding_idx=0)
# embedding.weight = nn.Parameter(embedding_matrix)
# embedding.weight.requires_grad = False  # Freeze the embedding layer
input = torch.LongTensor([[ 14588]])

# input[input > 107311] = 1
# print(input)

print(embedding(input))
print('embedding(input)')


##107312 >> out of rang

# To check the size of the vocabulary
# vocab_size = len(item2vec_model.wv)
# print("Vocabulary size:", vocab_size) #== 107312