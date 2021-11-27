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


class ICASSP3CNN(nn.Module):
    def __init__(self, embed_size=640, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional
        
        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(embed_size, embed_size, kernel_size=7, padding=3)

        self.batchnorm = nn.BatchNorm1d(3 * embed_size)

        self.lstm = nn.LSTM(input_size = 3 * embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """
        
        batch_size = x.shape[0]
        
        x = x.permute(1, 2, 0)    # (seq_len, batch_size, input_size) -> (batch_size, input_size, seq_len)
      
        cnn_output = torch.cat([self.cnn(x), self.cnn2(x), self.cnn3(x)], dim=1)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
        
        
class ICASSP2CNN(nn.Module):
    def __init__(self, embed_size=640, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)

        self.batchnorm = nn.BatchNorm1d(2 * embed_size)

        self.lstm = nn.LSTM(input_size = 2 * embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """
        
        batch_size = x.shape[0]
        # torch.Size([468, 64, 640])
        x = x.permute(1, 2, 0)    # (seq_len, batch_size, input_size) -> (batch_size, input_size, seq_len)

        cnn_output = torch.cat([self.cnn(x), self.cnn2(x)], dim=1)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
    

class ICASSP1CNN(nn.Module):
    def __init__(self, embed_size=640, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)

        self.batchnorm = nn.BatchNorm1d(embed_size)

        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """
        batch_size = x.shape[0]
     
        x = x.permute(1, 2, 0)    # (seq_len, batch_size, input_size) -> (batch_size, input_size, seq_len)

        cnn_output = self.cnn(x)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
        