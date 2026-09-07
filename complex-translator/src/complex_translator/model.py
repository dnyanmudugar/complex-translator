import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        # Combine forward and backward states cleanly
        hidden = torch.tanh(self.fc(torch.cat((hidden[0:1], hidden[1:2]), dim=2)))
        return outputs, hidden

class LuongAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(LuongAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, hidden, encoder_outputs):
        # input_step shape: (1, 1) representing the single current token index
        embedded = self.embedding(input_step)
        embedded = self.dropout(embedded)
        
        decoder_output, hidden = self.gru(embedded, hidden)
        
        src_len = encoder_outputs.size(1)
        attn_scores = torch.zeros(1, src_len, device=input_step.device)
        
        # Calculate alignment energy scores 
        for t in range(src_len):
            attn_scores[0, t] = torch.sum(decoder_output * encoder_outputs[0, t, :self.hidden_size])
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1) 
        context = torch.bmm(attn_weights, encoder_outputs) 
        
        concat_input = torch.cat((decoder_output, context), dim=2)
        concat_output = torch.tanh(self.concat(concat_input))
        
        output = self.out(concat_output)
        return output, hidden, attn_weights
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
