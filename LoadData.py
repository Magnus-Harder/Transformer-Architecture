#%%
import collections

from helper import load_data
import numpy as np
import torchtext


from torchtext.datasets import IWSLT2017
train_iter, valid_iter, test_iter = IWSLT2017()
src_sentence, tgt_sentence = next(iter(train_iter))

# %%
