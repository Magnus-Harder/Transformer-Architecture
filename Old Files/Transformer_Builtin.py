import torch as th
from torch import nn
from torch.nn import functional as F

# Define Embedder class
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

# Define function to create src_mask
def src_mask(D,n_pads):
    mask = th.full((D,D),float(-1e9))
    mask[:D-n_pads,:D-n_pads] = 0
    return mask

# Define function to create tgt_mask
def tgt_mask(D,n_pads):
    mask = th.full((D,D),float('-inf'))
    mask = mask.triu(diagonal=1)
    return mask

# Initiate Transformer model
class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,nhead):
        super().__init__()
        self.Transformer = nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=100,dropout=0.1,activation='relu',batch_first=True)
        self.Linear = nn.Linear(d_model,tgt_vocab_size)


        self.src_embed = Embedder(src_vocab_size, d_model)
        self.tgt_embed = Embedder(tgt_vocab_size, d_model)
        self.Sotftmax  = nn.Softmax(dim=1)



    def forward(self, X,tgt,src_mask=None,tgt_mask=None,src_key_padding_mask=None,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        X = self.src_embed(X)
        tgt = self.tgt_embed(tgt)
        
        out = self.Transformer(X,tgt,src_mask = src_mask ,
                                tgt_mask = tgt_mask,
                                src_key_padding_mask = src_key_padding_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask)
        out = self.Linear(out)
        out = self.Sotftmax(out)
        return out

# Define function to create src_mask
def src_mask(D,n_pads):
    mask = th.full((D,D),float(-1e9))
    mask[:D-n_pads,:D-n_pads] = 0
    return mask

# Define function to create tgt_mask
def tgt_mask(D,n_pads):
    mask = th.full((D,D),float('-inf'))
    mask = mask.triu(diagonal=1)
    return mask

# Define function to create src_key_padding_mask
def src_key_padding_mask(X):
    mask = th.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 0:
                mask[i,j] = 1
    return mask

# Define function to create tgt_key_padding_mask
def tgt_key_padding_mask(X):
    mask = th.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 0:
                mask[i,j] = 1
    return mask

# Define function to create memory_key_padding_mask
def memory_key_padding_mask(X):
    mask = th.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 0:
                mask[i,j] = 1
    return mask

