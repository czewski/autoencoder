import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embeddings = torch.load('embeddings.pt').cpu().numpy()
print(f"Embeddings shape: {embeddings.shape}")

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# Reduce dimensionality to 50 components using PCA first
# pca = PCA(n_components=50)
# embeddings_reduced = pca.fit_transform(embeddings)
# Apply t-SNE on the reduced embeddings
#embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings_reduced)

# Scatter plot with labels
# labels = np.array([0, 1, 2, ...])
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', s=10, alpha=0.7) #c=labels
plt.title('t-SNE Visualization of Embeddings with Labels')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(scatter)
plt.grid(True)
plt.show()