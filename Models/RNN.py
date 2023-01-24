# Importing the libraries

import torch as tn
import torch.nn as nn
import torch.nn.functional as F




# Define GRU Model class
class GRUNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_sequence, dim_output,src_padding_idx,src_vocab_size,tgt_vocab_size): 
        super().__init__()

        self.num_sequence = num_sequence
        self.Embedding_src = nn.Embedding(src_vocab_size,dim_input, padding_idx=src_padding_idx)
        #self.Softmax = nn.Softmax(dim = 1)

        self.Encoder = nn.GRU(input_size = dim_input,
                          hidden_size = dim_recurrent,
                          num_layers = num_sequence,
                          batch_first = True,
                          dropout = 0.1)

        self.Decoder = nn.GRU(input_size = dim_input,
                          hidden_size = dim_recurrent,
                          num_layers = 1,
                          batch_first = True,
                          dropout = 0.1)


        self.Forward = nn.Linear(in_features=dim_recurrent,out_features=tgt_vocab_size)


    def forward(self, input, predict_token):

        input = self.Embedding_src(input)

        predict_start = self.Embedding_src(predict_token)

        # Get the last layer's last time step activation
        output, hidden = self.Encoder(input)
        
        h0 = hidden[:][0]
        
        output[:][0] = predict_start

        for i in range(self.num_sequence):
            output_layer,h0 = self.Decoder(output[:][i],h0)
            output[:][i+1] = output_layer
 
        output = self.Forward(F.relu(output))
        
        return output