import torch.nn as nn
import torch.nn.functional as F
import torch

class KNNHelper:
    def __init__(self, distance_metric='euclidean'):
        self.distance_metric = distance_metric

    def compute_distance(self, query_embedding, data_embeddings):
        """
        Computes the distance between the query embedding and all data embeddings.
        """
        if self.distance_metric == 'euclidean':
            dists = torch.cdist(query_embedding.unsqueeze(0), data_embeddings)
        elif self.distance_metric == 'cosine':
            dists = 1 - F.cosine_similarity(query_embedding.unsqueeze(0), data_embeddings, dim=-1)
        return dists.squeeze(0)

    def find_closest_tensor(self, query_embedding, data_embeddings):
        """
        Finds the closest tensor in the dataset based on the distance metric.
        Returns the closest tensor and its index.
        """
        dists = self.compute_distance(query_embedding, data_embeddings)
        min_dist, closest_index = torch.min(dists, dim=0)
        closest_tensor = data_embeddings[closest_index]
        return closest_tensor, closest_index