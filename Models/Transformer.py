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

        # Define embedding layer
        self.embed = nn.Embedding(vocab_size, d_model,padding_idx)

    def forward(self, x):
        return self.embed(x)

# Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, T, d_model):
        super().__init__()

        # Define needed built in functions and parameters
        self.d_model = d_model
        self.T = T
        self.PositionalEncoding = nn.Parameter(th.zeros((int(T), int(d_model))), requires_grad=False)
        self.init_positional_encoding()

    # Initialize positional encoding matrix
    def init_positional_encoding(self):
        position = th.arange(0, self.T, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, self.d_model, 2,dtype=th.float) * (-th.log(th.tensor([10000.0])) / self.d_model))
        self.PositionalEncoding[:, 0::2] = th.sin(position * div_term)
        self.PositionalEncoding[:, 1::2] = th.cos(position * div_term)

    # Add positional encoding to input
    def forward(self, X):
        return X + self.PositionalEncoding



# Define Attention class
class Attention(nn.Module):
    def __init__(self, dk,dropout=0.1):
        super().__init__()

        # Define needed built in functions
        self.Softmax = nn.Softmax(dim=1)
        self.sqrt_dk = math.sqrt(dk)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,Q,K,V,mask=None):
    
        # Query key dot product
        QK = Q @ th.transpose(K,1,2) / self.sqrt_dk
        QK = self.dropout(QK)

        # Apply mask if not None
        if mask is not None:
            QK = QK + mask

        # Get attention weights
        A = th.transpose(self.Softmax(th.transpose(QK,1,2)),1,2) @ V

        return A


# Define multi head attention
class MHA(nn.Module):
    def __init__(self,T, d_model,dk=256,dv=512, nhead = 8,dropout=0.1):
        super().__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.nhead = nhead
        self.dk = dk
        self.dv = dv
        self.T = T
        
        # Define Q,K,V 
        self.Qs = nn.Linear(d_model,int(dk*nhead))
        self.Ks = nn.Linear(d_model,int(dk*nhead))
        self.Vs = nn.Linear(d_model,int(dv*nhead))

        # Define attention layer
        self.Attention = Attention(dk,dropout)

        # Define output layer    
        self.out = nn.Linear(int(dv*nhead),d_model)

    def forward(self,Q,K,V,mask=None):

        # Intialize Q,K,V
        Qs = self.Qs(Q)
        Ks = self.Ks(K)
        Vs = self.Vs(V)

        # Reshape Q,K,V to batch heads
        Qs = th.transpose(Qs.reshape(self.nhead,self.dk,self.T),1,2)
        Ks = th.transpose(Ks.reshape(self.nhead,self.dk,self.T),1,2)
        Vs = th.transpose(Vs.reshape(self.nhead,self.dv,self.T),1,2)

        # Get each attention heads
        A = self.Attention(Qs,Ks,Vs,mask)
        A = th.transpose(A,1,-1).reshape(-1,self.T).T        
        
        # Apply linear layer to get input dimensions
        return self.out(A)


# Define Feedforward class
class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.Linear1 = nn.Linear(d_model,d_ff)
        self.Linear2 = nn.Linear(d_ff,d_model)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
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

        # Define needed layers
        self.MHA = MHA(T, d_model, dk, dv, nhead)
        self.LayerNorm1 = LayerNorm(d_model)
        self.Feedforward = Feedforward(d_model, d_ff)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,X,mask=None):
        
        # Get multi head attention
        mha = self.MHA(X,X,X,mask)

        # Add residual and normalize
        X = self.LayerNorm1(self.dropout1(mha) + X)

        # Get feedforward
        X = self.Feedforward(X)

        # Add residual and normalize
        memory = self.LayerNorm2(X + self.dropout2(X))

        return memory 

# Define Encoder class
class Encoder(nn.Module):
    def __init__(self, T, d_model, nhead, d_ff, num_layers,dk,dv, dropout=0.1):
        super().__init__()

        # Define encoder layers
        self.encoders = nn.ModuleList([EncoderLayer(T, d_model, nhead, d_ff, dk, dv, dropout) for _ in range(num_layers)])

    def forward(self, X, mask=None):

        # Pass through each encoder layer
        for encoder in self.encoders:
            X = encoder(X, mask)

        return X


# Define Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self,T, d_model,nhead, d_ff, dk, dv, dropout=0.1):
        super().__init__()

        # Define needed layers
        self.MHA1 = MHA(T=T,d_model = d_model,dk = dk, dv = dv,nhead = nhead)
        self.LayerNorm1 = LayerNorm(d_model)
        self.MHA2 = MHA(T=T,d_model = d_model,dk = dk, dv = dv,nhead = nhead)
        self.LayerNorm2 = LayerNorm(d_model)
        self.Feedforward = Feedforward(d_model, d_ff)
        self.LayerNorm3 = LayerNorm(d_model)

        # Define dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,memory,tgt,mask=None):
        
        # Get multi head attention for target
        mha1 = self.MHA1(tgt,tgt,tgt,mask)

        # Add residual and normalize
        X = self.LayerNorm1(self.dropout1(mha1) + tgt)

        # Get multi head attention for memory and target
        mha2 = self.MHA2(memory,memory,X,mask=None)

        # Add residual and normalize
        X = self.LayerNorm2(self.dropout2(mha2) + X)

        # Get feedforward and add residual and normalize
        X = self.LayerNorm3(self.dropout3(self.Feedforward(X)) + X)

        return X



# Define Decoder class
class Decoder(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, num_layers, dk, dv, dropout=0.1):
        super().__init__()

        # Define decoder layers
        self.decoders = nn.ModuleList([DecoderLayer(T,d_model, nhead, d_ff, dk, dv, dropout) for _ in range(num_layers)])
        
    def forward(self, memory, tgt, src_mask=None, tgt_mask=None, tgt_padding_mask=None):

        # Pass through each decoder layer
        for decoder in self.decoders:
            tgt = decoder(memory,tgt,src_mask)

        return tgt

# Define Transformer class
class Transformer(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, num_layers, dk , dv, src_vocab_size,tgt_vocab_size,src_padding_idx,tgt_padding_idx, dropout=0.1):
        super().__init__()

        # Get Embedding and Positional Encoding layers
        self.Embedding_src = nn.Embedding(src_vocab_size,d_model, padding_idx=src_padding_idx)
        self.Embedding_tgt = nn.Embedding(tgt_vocab_size,d_model, padding_idx=tgt_padding_idx)
        self.PositionalEncoding = PositionalEncoding(T,d_model)

        # Define linear output Layer
        self.Linear_out = nn.Linear(d_model,tgt_vocab_size)

        # Define encoder and decoder
        self.encoder = Encoder(T,d_model, nhead, d_ff, num_layers, dk, dv, dropout)
        self.decoder = Decoder(T,d_model, nhead, d_ff, num_layers, dk, dv, dropout)

    def forward(self, src, tgt, src_padding_mask=None, tgt_mask=None, tgt_padding_mask=None):
        
        # Get embedding and positional encoding for tgt and src
        X = self.PositionalEncoding(self.Embedding_src(src))
        tgt = self.PositionalEncoding(self.Embedding_tgt(tgt))

        # Apply padding mask
        with th.no_grad():
            X =  X * src_padding_mask
            tgt = tgt * tgt_padding_mask

        # Get encoder and decoder output
        memory = self.encoder(X,)
        output = self.decoder(tgt, memory, tgt_mask)

        # Get output
        out = self.Linear_out(output)

        return out

