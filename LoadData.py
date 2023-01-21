#%%
from helper import load_data
from torchtext.data.utils import get_tokenizer
from torchtext import vocab
# Load English data
english_sentences = load_data('data/small_vocab_en')
# Load French data
french_sentences = load_data('data/small_vocab_fr')

# Remove the last sentence from the dataset (it's incomplete)
english_sentences.pop()
french_sentences.pop()

# Tokenize the sentences
tokenizer = get_tokenizer('basic_english')

english_sentences = [tokenizer(sentence) for sentence in english_sentences]
french_sentences = [tokenizer(sentence) for sentence in french_sentences]

# Add the start and end tokens to the sentences
Transformer_prediction = [["<Start>"],["<Pad>"],["<End>"]]

# Define the vocabulary for the English and French datasets
Vocab_en = vocab.build_vocab_from_iterator(english_sentences + Transformer_prediction, min_freq = 1, special_first = True)
Vocab_fr = vocab.build_vocab_from_iterator(french_sentences + Transformer_prediction, min_freq = 1, special_first = True)

# Max length of the sentences
max_len = max([max([len(sentence) for sentence in english_sentences]),max([len(sentence) for sentence in french_sentences])]) + 2 # for Padding and End tokens


# Create the list of padded sentences
Paddings_en = []
Paddings_fr = []

# Add the start and end tokens to the sentences
for sentence_en,sentence_fr in zip(english_sentences,french_sentences):

    # Add the start and end tokens to the sentences
    sentence_en.insert(0,"<Start>")
    sentence_fr.insert(0,"<Start>")
    sentence_en.append("<End>")
    sentence_fr.append("<End>")

    # Pad the sentences
    if len(sentence_en) < max_len:
        Paddings_en.append(max_len-len(sentence_en))
        sentence_en += ["<Pad>"]*(max_len-len(sentence_en))
        
    if len(sentence_fr) < max_len:
        Paddings_fr.append(max_len-len(sentence_fr))
        sentence_fr += ["<Pad>"]*(max_len-len(sentence_fr))
        
# Encode the sentences according to the vocabulary
english_encodings = [Vocab_en.lookup_indices(sentence) for sentence in english_sentences]
french_encodings = [Vocab_fr.lookup_indices(sentence) for sentence in french_sentences]

#%%
import pickle as pl

# Save the data
with open('data/English_encodings.pkl', 'wb') as f:
    pl.dump([english_encodings,english_sentences,Paddings_en,Vocab_en], f)
with open('data/French_encodings.pkl', 'wb') as f:
    pl.dump([french_encodings,french_sentences,Paddings_fr,Vocab_fr], f)








# %%
