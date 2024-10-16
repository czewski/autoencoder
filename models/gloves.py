from glove import Corpus, Glove
from utils import dataset
import numpy as np

# diginetica -> 43099
# yoochoose -> 37485

# Load your dataset (already handled)
root = "data/yoochoose1_64/"
train, valid, test = dataset.load_data_narm(root, valid_portion=0.1, maxlen=15)

# Create the co-occurrence matrix (sessions are passed as the training data)
corpus = Corpus()
corpus.fit(train[0], window=10)  # `train[0]` contains the session data

# Train the GloVe model
embedding_dim = 50
glove_model = Glove(no_components=embedding_dim, learning_rate=0.05)
glove_model.fit(corpus.matrix, epochs=50, no_threads=4, verbose=True)

# Save the trained GloVe model
glove_model.add_dictionary(corpus.dictionary)
glove_model.save("embeddings/yoochoose1_64/glove.model")


# Load GloVe model
glove_model = Glove.load("embeddings/yoochoose1_64/glove.model")

# Handle missing item IDs
total_ids = set(range(37485))  # Assuming IDs range from 0 to 43098
trained_ids = set(int(item) for item in glove_model.dictionary.keys())
missing_ids = total_ids - trained_ids

# Add zero embeddings for missing IDs
for missing_id in missing_ids:
    zero_embedding = [0.0] * embedding_dim
    glove_model.dictionary[str(missing_id)] = len(glove_model.dictionary)
    glove_model.word_vectors = np.vstack([glove_model.word_vectors, zero_embedding])

# Save the updated model
glove_model.save("embeddings/yoochoose1_64/glove.model")