# Importing the libraries

import torch as tn
import torch.nn as nn
import torch.nn.functional as F



# Define LSTM Model class
class LSTMNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output,src_padding_idx,src_vocab_size):
        super().__init__()

        self.Embedding_src = nn.Embedding(src_vocab_size,dim_input, padding_idx=src_padding_idx)



        self.lstm = nn.LSTM(input_size = dim_input,
                            hidden_size = dim_recurrent,
                            num_layers = num_layers,
                            batch_first = True)
        self.fc_o2y = nn.Linear(dim_recurrent, dim_output)
    def forward(self, input):

        input = self.Embedding_src(input)
        
        # Get the last layer's last time step activation
        output, _ = self.lstm(input)
        #output = output[-1]
        return self.fc_o2y(F.relu(output))

# Define GRU Model class
class GRUNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output,src_padding_idx,src_vocab_size): 
        super().__init__()

        self.Embedding_src = nn.Embedding(src_vocab_size,dim_input, padding_idx=src_padding_idx)
        #self.Softmax = nn.Softmax(dim = 1)

        self.gru = nn.GRU(input_size = dim_input,
                          hidden_size = dim_recurrent,
                          num_layers = num_layers,
                          batch_first = True)
        self.fc_y = nn.Linear(dim_recurrent, dim_output)
    def forward(self, input):

        input = self.Embedding_src(input)
        # Get the last layer's last time step activation
        output, _ = self.gru(input)
        output = self.fc_y(F.relu(output))
        #output = output[-1]
        return output