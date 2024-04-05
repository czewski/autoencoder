import torch

'''
Reference: https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
'''

def collate_fn(data):
    """This function will be used to pad the sessions to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors and lengths of each session (before padding).
       It will be used in the Dataloader.
    """
    data.sort(key=lambda x: len(x), reverse=True)
    lens = [len(sess) for sess in data]
    # 50x19
    padded_sesss = torch.zeros(len(data),20).long()
    for i, sess in enumerate(data): 
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
    
    #padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, lens

def create_matrix_mask(rows, max_size, x):
    if max(x) > max_size:
        raise ValueError("Maximum value in 'x' cannot exceed max_size.")
    
    mask = torch.zeros((rows, max_size), dtype=torch.int, device="cuda")
    for i in range(rows):
        row_size = min(max_size, x[i])
        mask[i, :row_size] = 1
    
    return mask