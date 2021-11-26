import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseLSTM(nn.Module):
    def __init__(self, num_layers, num_classes, input_size, hidden_size, dropout, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)

        self.linear = nn.Linear(in_features=2 * hidden_size if bidirectional else hidden_size, 
                                out_features=num_classes)
        
    def forward(self, x, lengths=None):
        batch_size = x.size(0)

        _, (hn, cn) = self.lstm(x)

        if self.bidirectional:
            h_n = hn.view(self.num_layers, 2, batch_size, self.hidden_size)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits