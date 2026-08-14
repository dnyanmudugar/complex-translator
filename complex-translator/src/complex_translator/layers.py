import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    Processes the input source sequence and creates hidden states.
    Uses an Embedding layer followed by a Gated Recurrent Unit (GRU).
    """
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, input_seq, hidden=None):
        # input_seq shape: (batch_size, sequence_length)
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, hidden_dim)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden


class LuongAttention(nn.Module):
    """
    Calculates attention weights using the 'Dot' score function:
    score(h_t, h_s) = h_t^T * h_s
    """
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch_size, hidden_dim) -> squeeze to (batch_size, hidden_dim, 1)
        tgt_state = decoder_hidden.transpose(0, 1).transpose(1, 2)
        
        # Matrix multiplication to find raw scores
        # (batch_size, seq_len, hidden_dim) x (batch_size, hidden_dim, 1) -> (batch_size, seq_len, 1)
        attn_scores = torch.bmm(encoder_outputs, tgt_state)
        
        # Softmax over sequence length gives the distribution weights
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Context vector is weighted sum of encoder states
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        return context, attn_weights


class AttnDecoderRNN(nn.Module):
    """
    Generates target tokens using past target inputs, encoder context, and attention.
    """
    def __init__(self, hidden_dim: int, output_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, hidden_dim)
        self.attention = LuongAttention()
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token shape: (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch_size, 1, hidden_dim)
        
        # Compute context using the current hidden state and encoder states
        context, attn_weights = self.attention(hidden, encoder_outputs)
        
        # Combine current embedding vector with history context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(output.squeeze(1))  # Project to vocabulary size
        
        return output, hidden, attn_weights
