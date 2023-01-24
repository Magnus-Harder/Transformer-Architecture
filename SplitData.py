#%%
import torch as th
from torchtext import vocab
import pickle as pl



# Load English data and French data
with open('data/English_encodings.pkl', 'rb') as f:
    english_encodings,english_sentences,Paddings_en,Vocab_en = pl.load(f)
with open('data/French_encodings.pkl', 'rb') as f:
    french_encodings,french_sentences,Paddings_fr,Vocab_fr = pl.load(f)

# Get the vocabulary size
src_vocab_size = Vocab_fr.__len__()
tgt_vocab_size = Vocab_en.__len__()

d_model = 512
# Define Train/test split and Masking
X = th.tensor(french_encodings)
Y = th.tensor(english_encodings)


n_train = 10000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_vali = X[10001:12000]
Y_vali = Y[10001:12000]
X_test = X[12001:]
Y_test = Y[12001:]

# Create batches
batch_size = 50

# Initialize the Batch tensors
X_train_batches = th.zeros((int(n_train/batch_size),batch_size,27),dtype = th.int64)
Y_train_batches = th.zeros((int(n_train/batch_size),batch_size,27),dtype = th.int64)

# Create the batches
for batch in range(int(n_train/batch_size)):

    # Fill Data batches
    X_train_batches[batch] = X_train[batch*batch_size:(batch+1)*batch_size]
    Y_train_batches[batch] = Y_train[batch*batch_size:(batch+1)*batch_size]

# Initialize the Mask tensors
#src_mask_test = tn.zeros((int(n_train/batch_size),batch_size*8,27,27))
#tgt_mask_test = tn.zeros((int(n_train/batch_size),batch_size*8,27,27))

src_key_masks = th.ones((int(n_train/batch_size),batch_size,27,d_model))
tgt_key_masks = th.ones((int(n_train/batch_size),batch_size,27,d_model))

idx_sample = 0
for batch in range(int(n_train/batch_size)):
    for sample in range(batch_size):
        
        src_key_masks[batch,sample][-Paddings_fr[idx_sample]:] = 0
        tgt_key_masks[batch,sample][-Paddings_en[idx_sample]:] = 0

        idx_sample += 1

# Validation Masks
src_key_masks_vali = th.ones(len(X_vali),27,d_model)

for sample in range(len(X_vali)):
    src_key_masks_vali[sample][-Paddings_fr[sample+n_train+1]:] = 0

# Test Masks
src_key_masks_test = th.ones(len(X_test),27,d_model)

for sample in range(len(X_test)):
    src_key_masks_test[sample][-Paddings_fr[sample+n_train+1+len(X_vali)]:] = 0

with open('data/Train_data.pkl', 'wb') as f:
    pl.dump([X_train_batches,Y_train_batches,src_key_masks,tgt_key_masks],f)

with open('data/Validation_data.pkl', 'wb') as f:
    pl.dump([X_vali,Y_vali,src_key_masks_vali],f)

with open('data/Test_data.pkl', 'wb') as f:
    pl.dump([X_test,Y_test,src_key_masks_test],f)

