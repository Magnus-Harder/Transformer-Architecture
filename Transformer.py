#%%
import torch as tn
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self,T,D):
        super().__init__()
        self.mht_d1 = nn.MultiheadAttention((T,D),10)
        self.lnorm = nn.LayerNorm((T,D))


        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
    def forward(self, X):
        X = self.mht_d1(X)
        return x


# %%

X = tn.ones((100,100,10))

m = nn.Linear(10,out_features=20,bias=False)

m(X)
# %%
nn.MultiheadAttention()

nn.Transformer()


# %%
