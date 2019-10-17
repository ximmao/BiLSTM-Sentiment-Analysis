import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class biLSTM_MLP(nn.Module):
    def __init__(self, embedding_dim, batch_size, vocab_size, hidden_dim_lstm, output_dim, use_bidirectional):
        super(biLSTM_MLP, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.use_bidirectional = use_bidirectional
        self.WordEmbedding_default = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_sgl = nn.LSTM(embedding_dim, hidden_dim_lstm, bidirectional=use_bidirectional)

        self.fc1 = nn.Linear(hidden_dim_lstm*2, 1024)
        self.fc2 = nn.Linear(1024,output_dim)
        self.dropout = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_dim_lstm*2, 1024)
        self.fc4 = nn.Linear(1024, output_dim)

    def forward(self, input):
        wordEmbed = self.WordEmbedding_default(input)
        batch_size = wordEmbed.size()[1]
        out, (hidden, cell) = self.lstm_sgl(wordEmbed, self.initHidden(batch_size))

        out_avg = torch.mean(out, 0)
        fc3_out = F.leaky_relu(self.fc3(out_avg), 0.2)
        fc3_dropout = self.dropout(fc3_out)
        return self.fc4(fc3_dropout)

    def initHidden(self, batch_size):
        if torch.cuda.is_available():
            if self.use_bidirectional:
                return (torch.zeros(2, batch_size, self.hidden_dim_lstm).cuda(), torch.zeros(2, batch_size, self.hidden_dim_lstm).cuda())
            else:
                return (torch.zeros(1, batch_size, self.hidden_dim_lstm).cuda(), torch.zeros(1, batch_size, self.hidden_dim_lstm).cuda())
        else:
            if self.use_bidirectional:
                return (torch.zeros(2, batch_size, self.hidden_dim_lstm), torch.zeros(2, batch_size, self.hidden_dim_lstm))
            else:
                return (torch.zeros(1, batch_size, self.hidden_dim_lstm), torch.zeros(1, batch_size, self.hidden_dim_lstm))
