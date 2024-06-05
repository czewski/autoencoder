import torch

def get_hit(indices, targets):
    """
    Calculates the HIT@K score for the given predictions and targets.

    Args:
        indices (BxK): torch.LongTensor. Top-K indices predicted by the model.
        targets (B): torch.LongTensor. Actual target indices.

    Returns:
        hit (int): 1 if there is at least one hit, 0 otherwise.
    """
    # Check if any of the true targets are in the top-K recommendations
    hit = any(target in indices[i] for i, target in enumerate(targets))
    
    # Convert boolean to integer (1 for True, 0 for False)
    return int(hit)


def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()


def evaluate(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K, and HIT@K scores.

    Args:
        indices (B,C): torch.LongTensor. The predicted indices for the next items.
        targets (B): torch.LongTensor. actual target indices.
        k (int): The number of top recommendations to consider.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
        hit (int): 1 if there is at least one hit, 0 otherwise.
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    hit = get_hit(indices, targets)
    return recall, mrr, hit
