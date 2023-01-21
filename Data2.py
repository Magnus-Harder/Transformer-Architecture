#%%
import torch as tn
from torchtext import vocab
import pickle as pl
from Transformer import Transformer,src_mask,tgt_mask
from tqdm import tqdm
import numpy as np

# Load English data and French data
with open('data/English_encodings.pkl', 'rb') as f:
    english_encodings,english_sentences,Paddings_en,Vocab_en = pl.load(f)
with open('data/French_encodings.pkl', 'rb') as f:
    french_encodings,french_sentences,Paddings_fr,Vocab_fr = pl.load(f)

# Get the vocabulary size
src_vocab_size = Vocab_fr.__len__()
tgt_vocab_size = Vocab_en.__len__()
#%%

X = tn.tensor(french_encodings)
Y = tn.tensor(english_encodings)


n_train = 1000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_vali = X[10001:12000]
Y_vali = Y[10001:12000]
X_test = X[12001:]
Y_test = Y[12001:]

# Create batches
batch_size = 100 

# Initialize the Batch tensors
X_train_batches = tn.zeros((int(n_train/batch_size),batch_size,27),dtype = tn.int64)
Y_train_batches = tn.zeros((int(n_train/batch_size),batch_size,27),dtype = tn.int64)

# Create the batches
for batch in range(int(n_train/batch_size)):

    # Fill Data batches
    X_train_batches[batch] = X_train[batch*batch_size:(batch+1)*batch_size]
    Y_train_batches[batch] = Y_train[batch*batch_size:(batch+1)*batch_size]

# Initialize the Mask tensors
#src_mask_test = tn.zeros((int(n_train/batch_size),batch_size*8,27,27))
#tgt_mask_test = tn.zeros((int(n_train/batch_size),batch_size*8,27,27))

src_key_masks = tn.zeros((int(n_train/batch_size),batch_size,27),dtype = tn.bool)
tgt_key_masks = tn.zeros((int(n_train/batch_size),batch_size,27),dtype=tn.bool)

idx_sample = 0
for batch in range(int(n_train/batch_size)):
    for sample in range(batch_size):
        
        src_key_masks[batch,sample][-Paddings_fr[idx_sample]:] = 1
        tgt_key_masks[batch,sample][-Paddings_en[idx_sample]:] = 1

        # for i in range(27):
        #     if np.random.rand() < 0.1:
        #         tgt_key_masks[batch,sample][i] = True

        #tgt_mask_test[batch,idx_sample*8:idx_sample*8+8] = tgt_mask(27,Paddings_en[idx_sample])
        #src_mask_test[batch,idx_sample*8:idx_sample*8+8] = src_mask(27,Paddings_fr[idx_sample])

        idx_sample += 1



no_ahead_mask = tn.triu(tn.full((27,27),float('-inf')),diagonal=1)

    

#%%
loss_fn = tn.nn.CrossEntropyLoss()
Model = Transformer(src_vocab_size,tgt_vocab_size,d_model=128,nhead=4)
optimizer = tn.optim.Adam(Model.parameters(), lr=0.01)



out = Model(X_train_batches[0][2],Y_train_batches[0][1],tgt_mask = no_ahead_mask,src_key_padding_mask = src_key_masks[0][0],tgt_key_padding_mask = tgt_key_masks[0][0])


pred = out.argmax(1).tolist()
Sentece = Vocab_en.lookup_tokens(pred)
Sentece
#%%

def trainmodel(epochs):
    if tn.cuda.is_available():
        device = tn.device("cuda")
    elif tn.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    Model.to(device)
    X_train_batches_mps = X_train_batches.to(device)
    Y_train_batches_mps = Y_train_batches.to(device)
    no_ahead_mask_mps = no_ahead_mask.to(device)


    for epoch in tqdm(range(epochs)):
        loss_epoch = 0

        for X_batch,Y_batch,src_m,tgt_m in zip(X_train_batches_mps,Y_train_batches_mps,src_key_masks.to(device),tgt_key_masks.to(device)):

            optimizer.zero_grad()
            out = Model(X_batch,Y_batch,
                        tgt_mask = no_ahead_mask_mps,
                        src_key_padding_mask = src_m,
                        tgt_key_padding_mask = tgt_m
                        )

            loss = 0
            for i in range(batch_size):
                #index = [Y_batch[i] != 205]
                #loss += loss_fn(out[i][index],Y_batch[i][index])
                loss += loss_fn(out[i],Y_batch[i])
            #loss = loss_fn(out,Y_batch)
            loss_epoch += loss.item()

            loss.backward()
            
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {loss_epoch/n_train}")    

#%%

trainmodel(100)

tn.save(Model.state_dict(), "Transformer.pt")
        
#%%
Pred_mask = tn.tensor([False]+[True for i in range(26)],dtype=tn.bool)
Model.to("cpu")
out = Model(X_train_batches[0][0],Y_train_batches[0][0],tgt_mask = no_ahead_mask,src_key_padding_mask = src_key_masks[0][0],tgt_key_padding_mask = Pred_mask)
pred = out.argmax(1).tolist()
Sentece = Vocab_en.lookup_tokens(pred)
print(Sentece)

#%%

src_mask_key = tn.zeros(27,dtype=tn.bool)
src_mask_key[-Paddings_fr[-1]:] = 1


Pred_mask = tn.tensor([True for i in range(27)],dtype=tn.bool)


st_input = tn.tensor([206]+[205 for i in range(26)])




for i in range(26):

    Pred_mask[i] = False

    out = Model(X_test[-1],
                #Y_test[-1],
                st_input,
                src_mask = None,
                tgt_mask = no_ahead_mask,

                src_key_padding_mask = src_mask_key,
                tgt_key_padding_mask = Pred_mask
                )
    pred = out.argmax(1).tolist()
    st_input[i+1] = pred[i+1]

Sentece = Vocab_en.lookup_tokens(st_input.tolist())
print(Sentece)


# #%%
# import torch as tn

# src_vocab_size = Vocab_fr.__len__()+1
# tgt_vocab_size = Vocab_en.__len__()

# model = Transformer(src_vocab_size,tgt_vocab_size,d_model=512 ,nhead=8)
# model.load_state_dict(tn.load('Transformer.pt',map_location=tn.device('cpu')))
# model.eval()

# #%%

# st_input = tn.tensor([206]+[205 for i in range(26)])

# out = model(X_test[-8],st_input)
# pred = out.argmax(1).tolist()
# Sentece = Vocab_en.lookup_tokens(pred)
# Sentece

#%%
# for X_batch,Y_batch in zip(X_train_batches,Y_train_batches):
#             out = Model(X_batch,Y_batch)



# %%
