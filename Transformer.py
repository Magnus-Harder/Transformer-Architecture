#%%
import torch as tn
from torch import nn
from torch.nn import functional as F




# Embed(Vocabsize, embeddim) - 
# 


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Transformer = nn.Transformer(d_model=300,nhead=10,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,dropout=0.1,activation='relu')
        self.Linear = nn.Linear(25*300,204)
        self.Softmax = nn.Softmax(dim=204)


    def forward(self, X):
        X = self.Transformer(X)
        X.reshape(-1,25*300)
        X = self.Linear(X)
        X = self.Softmax(X)




        return X



# %%
