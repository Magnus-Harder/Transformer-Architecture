#%%
import torch as tn
from torchtext import vocab
import pickle


from helper import load_data
# Load English data
english_sentences = load_data('data/small_vocab_en')
# Load French data
french_sentences = load_data('data/small_vocab_fr')

english_sentences.pop()
french_sentences.pop()
#%%
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')


english_sentences = [tokenizer(sentence) for sentence in english_sentences]
french_sentences = [tokenizer(sentence) for sentence in french_sentences]
#%%
Transformer_prediction = [["<Start>"],["<Pad>"],["<End>"]]

All_tokens = Transformer_prediction + english_sentences + french_sentences

Vocab_en = vocab.build_vocab_from_iterator(english_sentences + Transformer_prediction, min_freq = 1, special_first = True)
Vocab_fr = vocab.build_vocab_from_iterator(french_sentences + Transformer_prediction, min_freq = 1, special_first = True)


#%%

english_encodings = [Vocab_en.lookup_indices(sentence) for sentence in english_sentences]
french_encodings = [Vocab_fr.lookup_indices(sentence) for sentence in french_sentences]


max = 0 
for sentence_en,sentence_fr in zip(english_encodings,french_encodings):
    if len(sentence_en) > max:
        max = len(sentence_en)
    if len(sentence_fr) > max:
        max = len(sentence_fr)

max_len = max + 2
idx = 0
for sentence_en,sentence_fr in zip(english_encodings,french_encodings):
    sentence_en.insert(0,int(Vocab_en.lookup_indices(["<Start>"])[0]))
    sentence_fr.insert(0,int(Vocab_fr.lookup_indices(["<Start>"])[0]))
    sentence_en.append(int(Vocab_en.lookup_indices(["<End>"])[0]))
    sentence_fr.append(int(Vocab_fr.lookup_indices(["<End>"])[0]))



    if len(sentence_en) < max_len:
        sentence_en += [int(Vocab_en.lookup_indices(["<Pad>"])[0])]*(max_len-len(sentence_en))
    if len(sentence_fr) < max_len:
        sentence_fr += [int(Vocab_fr.lookup_indices(["<Pad>"])[0])]*(max_len-len(sentence_fr))

    english_encodings[idx] = sentence_en
    french_encodings[idx] = sentence_fr
    idx += 1



#%%

from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,nhead):
        super().__init__()
        self.Transformer = nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,dropout=0.1,activation='relu')
        self.Linear = nn.Linear(d_model,tgt_vocab_size)


        self.src_embed = Embedder(src_vocab_size, d_model)
        self.tgt_embed = Embedder(tgt_vocab_size, d_model)
        self.Sotftmax  = nn.Softmax(dim=1)



    def forward(self, X,tgt):
        X = self.src_embed(X)
        tgt = self.tgt_embed(tgt)
        
        out = self.Transformer(X,tgt)
        out = self.Linear(out)
        out = self.Sotftmax(out)
        return out


src_vocab_size = Vocab_fr.__len__()
tgt_vocab_size = Vocab_en.__len__()
#%%

X = tn.tensor(french_encodings)
Y = tn.tensor(english_encodings)

X_train = X[:10000]
Y_train = Y[:10000]
X_vali = X[10001:12000]
Y_vali = Y[10001:12000]
X_test = X[12001:]
Y_test = Y[12001:]


batch_size = 50 
X_train_batches = tn.zeros((int(10000/batch_size),batch_size,27),dtype = tn.int64)
Y_train_batches = tn.zeros((int(10000/batch_size),batch_size,27),dtype = tn.int64)

for batch in range(200):
    X_train_batches[batch] = X_train[batch*batch_size:(batch+1)*batch_size]
    Y_train_batches[batch] = Y_train[batch*batch_size:(batch+1)*batch_size]


#%%
loss_fn = tn.nn.CrossEntropyLoss()
Model = Transformer(src_vocab_size,tgt_vocab_size,d_model=512,nhead=8)

out = Model(X_train[0],Y_train[0])
pred = out.argmax(dim=1).tolist()
Sentece = Vocab_en.lookup_tokens(pred)
print(Sentece)
loss_fn(out,Y_train[0])


loss_fn = tn.nn.CrossEntropyLoss()
out_batch = Model(X_train_batches[0],Y_train_batches[0])

loss = 0
for i in range(50):
    loss += loss_fn(out_batch[i],Y_train_batches[0][i])




# %%
from tqdm import tqdm

def trainmodel(epochs):
    if tn.cuda.is_available():
        device = tn.device("cuda")
    elif tn.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"
    Model.to(device)
    X_train_batches.to(device)
    Y_train_batches.to(device)
    
    loss_fn = tn.nn.CrossEntropyLoss()
    optimizer = tn.optim.Adam(Model.parameters(), lr=0.0001)    


    for epoch in tqdm(range(epochs)):
        for X_batch,Y_batch in zip(X_train_batches,Y_train_batches):
            optimizer.zero_grad()
            out = Model(X_batch.long(),Y_batch.long())

            loss = 0
            for i in range(50):
                loss += loss_fn(out[i],Y_batch[i])
            #loss = loss_fn(out,Y_batch)
            loss.backward()
            
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {loss.item()}")    
#%%

trainmodel(2)

tn.save(Model.state_dict(), "Transformer.pt")
        

#%%
Model = tn.load("Transformer.pt")
Model.eval()

#%%
for X_batch,Y_batch in zip(X_train_batches,Y_train_batches):
            out = Model(X_batch,Y_batch)



# %%
