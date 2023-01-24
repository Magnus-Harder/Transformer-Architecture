# Importing the libraries

import torch as tn
import torch.nn as nn
import torch.nn.functional as F




# Define GRU Model class
class GRUNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_sequence, src_padding_idx,tgt_padding_idx,src_vocab_size,tgt_vocab_size): 
        super().__init__()

        self.num_sequence = num_sequence
        self.Embedding_src = nn.Embedding(src_vocab_size,dim_input, padding_idx=src_padding_idx)
        self.Embedding_tgt = nn.Embedding(tgt_vocab_size,dim_input, padding_idx=tgt_padding_idx)
        #self.Softmax = nn.Softmax(dim = 1)

        self.Encoder = nn.GRU(input_size = dim_input,
                          hidden_size = dim_recurrent,
                          num_layers = num_sequence,
                          batch_first = True,
                          dropout = 0.1)

        self.Decoder = nn.GRU(input_size = dim_input,
                          hidden_size = dim_recurrent,
                          num_layers = num_sequence,
                          batch_first = True,
                          dropout = 0.1)


        self.Forward = nn.Linear(in_features=dim_recurrent,out_features=tgt_vocab_size)


    def forward(self, input, predict_token,device = 'cpu'):
        
        word_prob = tn.zeros((input.shape[0],self.num_sequence,self.Forward.out_features),device=device)

        input = self.Embedding_src(input)

        predict_start = self.Embedding_src(predict_token)

        # Get the last layer's last time step activation
        _, hidden = self.Encoder(input)

        hidden= hidden.rep
        
        output_decoder = tn.zeros(input.shape,device=device)
        hidden_decoder = tn.zeros((input.shape[0],self.num_sequence,hidden.shape[2]),device=device)
        
        output_decoder[:,0] = predict_start.clone()
        hidden_decoder[:,0] = hidden[-1].clone()
    
        for i in range(self.num_sequence-1):
            hidden_decoder[:,i+1] = self.Decoder(output_decoder[:,i].clone(),hidden_decoder[:,i].clone())
            
            word_prob[:,i+1] = self.Forward(F.relu(hidden_decoder[:,i+1].clone()))

            output_decoder[:,i+1] = self.Embedding_tgt(word_prob[:,i+1].clone().argmax(dim=1))
 
        
        return word_prob