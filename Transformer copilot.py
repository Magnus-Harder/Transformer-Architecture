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


# Define PositionalEncoder class
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = th.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = th.sin(pos / (10000 ** ((2 * i)/d_model)))
                self.pe[pos, i + 1] = th.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.pe = self.pe.unsqueeze(0)
    def forward(self, x):
        x = x * th.sqrt(th.tensor(self.d_model, dtype=th.float))
        seq_len = x.size(1)
        x = x + th.tensor(self.pe[:, :seq_len], requires_grad=False)
        return x


# Define MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        bsz = query.size(0)
        tgt_len, src_len = query.size(1), key.size(1)
        qkv = self.qkv(query).view(bsz, tgt_len, 3, self.nhead, self.d_model // self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * th.sqrt(th.tensor(self.d_model // self.nhead, dtype=th.float))
        attn = (q @ k.transpose(-2, -1)) / th.sqrt(th.tensor(self.d_model, dtype=th.float))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        self.attn = attn
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(bsz, tgt_len, self.d_model)
        x = self.out(x)
        return x


# Define FeedForward class
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define EncoderLayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x2 = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

# Define DecoderLayer class
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.multihead_attn(x, memory, memory, src_mask)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.ff(x)
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x

# Define Encoder class
class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x

# Define Decoder class
class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return x

# Define Transformer class
