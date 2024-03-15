import torch
import time

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
    lens = [len(sess[0]) for sess in data]
    # 50x16 
    padded_sesss = torch.zeros(len(data),max(lens)).long()
    for i, sess in enumerate(data): 
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess[0])
    
    print(padded_sesss)
    return padded_sesss, lens