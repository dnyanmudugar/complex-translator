import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Converts token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Gated Recurrent Unit (GRU) for sequence processing
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, hidden):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x) # shape: (batch_size, seq_len, embedding_dim)
        output, hidden = self.gru(embedded, hidden)
        # output shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (1, batch_size, hidden_dim)
        return output, hidden

    def init_hidden(self, batch_size, device):
        # Generates a clean starting matrix of zeros for each training batch
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query, values):
        # query (decoder hidden state) shape: (1, batch_size, hidden_dim)
        # values (encoder outputs) shape: (batch_size, seq_len, hidden_dim)
        query_with_time_axis = query.transpose(0, 1) # shape: (batch_size, 1, hidden_dim)

        # score shape: (batch_size, seq_len, 1)
        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(score, dim=1)

        # context_vector shape: (batch_size, hidden_dim)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(hidden_dim)
        # GRU input size combines both embedding dimensions and attention context vector
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        # Linear layer mapping hidden states back to vocabulary distribution
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        # x shape: (batch_size, 1) -> input token at current time step
        # hidden shape: (1, batch_size, hidden_dim) -> previous decoder hidden state
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        
        # 1. Compute attention alignment and get the context vector
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)

        # 2. Pass target token through embedding layer
        embedded = self.embedding(x) # shape: (batch_size, 1, embedding_dim)

        # 3. Concatenate the embedded token with the context vector
        context_vector_expanded = context_vector.unsqueeze(1) # shape: (batch_size, 1, hidden_dim)
        predict_input = torch.cat((context_vector_expanded, embedded), dim=-1)

        # 4. Process combined vector with GRU
        output, hidden = self.gru(predict_input, hidden)

        # 5. Format tensor shapes and generate prediction scores across vocabulary
        output = output.squeeze(1) # shape: (batch_size, hidden_dim)
        predictions = self.fc(output) # shape: (batch_size, vocab_size)

        return predictions, hidden, attention_weights
