#%%
import torch as tn
import pickle




with open('data/X.pickle','rb') as f:
    X_train,X_vali,X_test = pickle.load(f)
with open('data/Y.pickle','rb') as f:
    Y_train,Y_vali,Y_test = pickle.load(f)

from torch.optim import Adam

#%%
import torch as tn
from torch import nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Transformer = nn.Transformer(d_model=300,nhead=5,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,dropout=0.1,activation='relu')
        self.Linear = nn.Linear(25*300,204)
        self.Softmax = nn.Softmax(0)


    def forward(self, X,tgt):
        X = self.Transformer(X,tgt)
        X = X.reshape(1,-1)
        X = self.Linear(X)
        X = X.flatten()
        X = self.Softmax(X)



        return X

loss_fn = tn.nn.CrossEntropyLoss()


Model = Transformer()

Model.mode='predict'
#%%
out = Model.forward(X_train[0],tn.zeros((1,300)))
    
# %%
