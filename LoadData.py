#%%

import torch as tn
from torchtext import vocab
import pickle


vec_GloVe = vocab.GloVe(name='840B', dim=300)

from helper import load_data
# Load English data
english_sentences = load_data('data/small_vocab_en')
# Load French data
french_sentences = load_data('data/small_vocab_fr')


#%%
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')

english_sentences = [tokenizer(sentence) for sentence in english_sentences]
french_sentences = [tokenizer(sentence) for sentence in french_sentences]

english_sentences.pop()
french_sentences.pop()

max = 0 
for sentence_en,sentence_fr in zip(english_sentences,french_sentences):
    if len(sentence_en) > max:
        max = len(sentence_en)
    if len(sentence_fr) > max:
        max = len(sentence_fr)

print(max)
#%%
N = len(english_sentences)

X = tn.zeros((N,25,300))
Y = tn.zeros((N,25,300))

idx =0
error_idx = []
for english_sentence, french_sentence in zip(english_sentences, french_sentences):
    
    n_sentence_en = len(english_sentence)
    n_sentence_fr = len(french_sentence)


    Y[idx,:n_sentence_en:] = vec_GloVe.get_vecs_by_tokens(english_sentence)
    X[idx,:n_sentence_fr,:] = vec_GloVe.get_vecs_by_tokens(french_sentence)

    idx += 1
#%%

X_train = X[:10000]
Y_train = Y[:10000]
X_vali = X[10001:12000]
Y_vali = Y[10001:12000]
X_test = X[12001:]
Y_test = Y[12001:]

#%%


with open('data/X.pickle','wb') as f:
    pickle.dump([X_train,X_vali,X_test],f)
with open('data/Y.pickle','wb') as f:
    pickle.dump([Y_train,Y_vali,Y_test],f)


#%%
Vocab = vocab.build_vocab_from_iterator(english_sentences, min_freq = 1, special_first = True)
Vocab_glove = vec_GloVe.get_vecs_by_tokens(vocab.get_itos())

with open('data/Vocab_glove.pickle','wb') as f:
    pickle.dump([Vocab,Vocab_glove],f) 

