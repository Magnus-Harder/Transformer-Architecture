#%%
import torch as th
from torch import nn
from torch.nn import functional as F
import math

#%%
# Define Embedder class
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model,padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model,padding_idx)
    def forward(self, x):
        return self.embed(x)

# Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, T, d_model):
        super().__init__()
        self.d_model = d_model
        self.T = T
        self.PositionalEncoding = nn.Parameter(th.zeros((int(T), int(d_model))), requires_grad=False)
        self.init_positional_encoding()

    def init_positional_encoding(self):
        position = th.arange(0, self.T, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, self.d_model, 2,dtype=th.float) * (-th.log(th.tensor([10000.0])) / self.d_model))
        self.PositionalEncoding[:, 0::2] = th.sin(position * div_term)
        self.PositionalEncoding[:, 1::2] = th.cos(position * div_term)

    def forward(self, X):
        return X + self.PositionalEncoding



# Define Attention class
class Attention(nn.Module):
    def __init__(self, dk):
        super().__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.sqrt_dk = math.sqrt(dk)
        
    def forward(self,Q,K,V,mask=None):
    
        # Måske if statement med mask
        QK = Q @ th.transpose(K,1,2) / self.sqrt_dk
        if mask is not None:
            QK = QK + mask
        A = th.transpose(self.Softmax(th.transpose(QK,1,2)),1,2) @ V

        return A


# Define multi head attention
class MHA(nn.Module):
    def __init__(self,T, d_model,dk=256,dv=512, nhead = 8):
        super().__init__()
        self.Softmax = nn.Softmax(dim=1)
       #self.d_model = d_model
        self.nhead = nhead
        self.dk = dk
        self.dv = dv
        self.T = T
        
        # Define Q,K,V 
        self.Qs = nn.Linear(d_model,int(dk*nhead))
        self.Ks = nn.Linear(d_model,int(dk*nhead))
        self.Vs = nn.Linear(d_model,int(dv*nhead))

        self.Attention = Attention(dk)

        # Define output layer    
        self.out = nn.Linear(int(dv*nhead),d_model)

    def forward(self,Q,K,V,mask=None):

        # Intialize Q,K,V
        Qs = self.Qs(Q)
        Ks = self.Ks(K)
        Vs = self.Vs(V)

  
        Qs = th.transpose(Qs.reshape(self.nhead,self.dk,self.T),1,2)
        Ks = th.transpose(Ks.reshape(self.nhead,self.dk,self.T),1,2)
        Vs = th.transpose(Vs.reshape(self.nhead,self.dv,self.T),1,2)

        # Get each attention head
        A = self.Attention(Qs,Ks,Vs,mask)
        A = th.transpose(A,1,-1).reshape(-1,self.T).T        
        
        return self.out(A)


# Define Feedforward class
class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.Linear1 = nn.Linear(d_model,d_ff)
        self.Linear2 = nn.Linear(d_ff,d_model)
        self.ReLU = nn.ReLU()
    def forward(self,X):
        return self.Linear2(self.ReLU(self.Linear1(X)))

# Define LayerNorm class
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        #self.d_model = d_model
        self.LayerNorm = nn.LayerNorm(d_model)
    def forward(self,X):
        return self.LayerNorm(X)


# Define Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, dk, dv, dropout=0.1):
        super().__init__()
        # self.d_model = d_model
        # self.d_ff = d_ff
        # self.dk = dk
        # self.dv = dv
        # self.nhead = nhead
        self.MHA = MHA(T, d_model, dk, dv, nhead)
        self.LayerNorm1 = LayerNorm(d_model)
        self.Feedforward = Feedforward(d_model, d_ff)
        self.LayerNorm2 = LayerNorm(d_model)
    def forward(self,X,mask=None):
        
        mha = self.MHA(X,X,X,mask)

        return self.LayerNorm2(mha + self.Feedforward(self.LayerNorm1(mha+X)))

# Define Encoder class
class Encoder(nn.Module):
    def __init__(self, T, d_model, nhead, d_ff, num_layers,dk,dv, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer(T, d_model, nhead, d_ff, dk, dv, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, X, mask=None):
        for encoder in self.encoders:
            X = encoder(X, mask)

        X = self.norm(X)
        return X


# Define Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self,T, d_model,nhead, d_ff, dk, dv, dropput=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dk = dk
        self.dv = dv
        self.nhead = nhead
        self.T = T
        self.MHA1 = MHA(T=self.T,d_model = self.d_model,dk = self.dk, dv = self.dv,nhead = self.nhead)
        self.LayerNorm1 = LayerNorm(d_model)
        self.MHA2 = MHA(T=self.T,d_model = self.d_model,dk = self.dk, dv = self.dv,nhead = self.nhead)
        self.LayerNorm2 = LayerNorm(d_model)
        self.Feedforward = Feedforward(d_model, d_ff)
        self.LayerNorm3 = LayerNorm(d_model)
    def forward(self,memory,tgt,mask=None):
        
        mha1 = self.MHA1(tgt,tgt,tgt,mask)
    
        mha2 = self.MHA2(memory,memory,mha1,mask=None)

        return self.LayerNorm3(mha2 + self.Feedforward(self.LayerNorm2(mha2+mha1)))



# Define Decoder class
class Decoder(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, num_layers, dk, dv, dropout=0.1):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(T,d_model, nhead, d_ff, dk, dv, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, memory, tgt, src_mask=None, tgt_mask=None, tgt_padding_mask=None):
        for decoder in self.decoders:
            tgt = decoder(memory,tgt,src_mask)
        Y = self.norm(tgt)
        return Y

# Define Transformer class
class Transformer(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, num_layers, dk , dv, src_vocab_size,tgt_vocab_size,src_padding_idx,tgt_padding_idx, dropout=0.1):
        super().__init__()
        self.Embedding_src = nn.Embedding(src_vocab_size,d_model, padding_idx=src_padding_idx)
        self.Embedding_tgt = nn.Embedding(tgt_vocab_size,d_model, padding_idx=tgt_padding_idx)
        self.PositionalEncoding = PositionalEncoding(T,d_model)
        self.Linear_out = nn.Linear(d_model,tgt_vocab_size)
        self.Sotftmax  = nn.Softmax(dim=1)


        self.encoder = Encoder(T,d_model, nhead, d_ff, num_layers, dk, dv, dropout)
        self.decoder = Decoder(T,d_model, nhead, d_ff, num_layers, dk, dv, dropout)
    def forward(self, src, tgt, src_padding_mask=None, tgt_mask=None, tgt_padding_mask=None):
        
        X = self.PositionalEncoding(self.Embedding_src(src))
        tgt = self.PositionalEncoding(self.Embedding_tgt(tgt))
        with th.no_grad():
            X =  X * src_padding_mask
            tgt = tgt * tgt_padding_mask

        memory = self.encoder(X,)
        output = self.decoder(tgt, memory, tgt_mask)

        out = self.Linear_out(output)
        out = self.Sotftmax(out)
        return out

# %%

# 3     3   0
# 2     2   0
# 0     0   0

