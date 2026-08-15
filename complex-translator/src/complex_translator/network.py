import torch
<<<<<<< HEAD
import os

# Import everything properly
from neural_machine_translation import Translator, Encoder, Decoder
from neural_machine_translation import MultilingualTokenizer

json_file = "vocab.json"
model_name = "multilingual_transformer.pt"

# Define the runtime device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load the tokenizer (This fixes the NameError!)
tokenizer = MultilingualTokenizer()

# Point to your vocabulary mapping file inside your project directory
vocab_path = json_file
tokenizer.load_vocab(vocab_path)

# Define dimensions and initialize your model layers
VOCAB_SIZE = len(tokenizer.token_to_id)
EMBED_DIM = 256
HIDDEN_DIM = 512

encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

# Load your trained model weights
checkpoint_path = model_name
if os.path.exists(checkpoint_path):
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model weights loaded successfully.")

# Initialize your pipeline using the lowercase instances (Line 13 Fix!)
translator = Translator((encoder, decoder), tokenizer, device)
print("Translator is fully online and ready!")
=======
import torch.nn as nn
import random

# Placeholder setup to test execution within network.py locally
from pipeline import MultilingualVocabulary
from device import device

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        output, hidden = self.gru(embedded)
        return output, hidden

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=20):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, encoder_hidden, encoder_outputs, target_tensor=None, max_len=20, teacher_forcing_ratio=0.5):
        batch_size = encoder_hidden.size(1) if encoder_hidden.dim() > 2 else encoder_hidden.size(0)
        
        # Enforce strict Long type indices on the correct hardware device
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs_list = []

        steps = target_tensor.size(1) if target_tensor is not None else max_len
        use_teacher_forcing = True if target_tensor is not None and random.random() < teacher_forcing_ratio else False

        # Pad encoder outputs dynamically to match max_length
        current_seq_len = encoder_outputs.size(1)
        if current_seq_len < self.max_length:
            padding = torch.zeros(batch_size, self.max_length - current_seq_len, self.hidden_size, device=device)
            encoder_outputs = torch.cat((encoder_outputs, padding), dim=1)
        else:
            encoder_outputs = encoder_outputs[:, :self.max_length, :]

        hidden_for_attn = decoder_hidden.squeeze(0) if decoder_hidden.dim() == 3 else decoder_hidden

        for i in range(steps):
            embedded = self.embedding(decoder_input)
            attn_weights = torch.softmax(self.attn(torch.cat((embedded[:, 0], hidden_for_attn), 1)), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            
            output = torch.cat((embedded[:, 0], context[:, 0]), 1)
            output = self.attn_combine(output).unsqueeze(1)
            output = torch.relu(output)
            
            output, decoder_hidden = self.gru(output, decoder_hidden)
            hidden_for_attn = decoder_hidden.squeeze(0) if decoder_hidden.dim() == 3 else decoder_hidden
            
            top_features = self.out(output)
            decoder_outputs_list.append(top_features)

            if use_teacher_forcing:
                decoder_input = target_tensor[:, i].unsqueeze(1).to(dtype=torch.long, device=device)
            else:
                _, topi = top_features.topk(1)
                decoder_input = topi.squeeze(-1).detach().to(dtype=torch.long, device=device)

        return torch.cat(decoder_outputs_list, dim=1), decoder_hidden, None

class Translator:
    # FIXED: Signature explicitly maps separate positional inputs instead of packed tuples
    def __init__(self, encoder, decoder, vocab, device=device):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device

    def translate(self, text, src_lang, tgt_lang):
        """Clean 3-argument translate method interface using keywords safely."""
        self.encoder.eval()
        self.decoder.eval()
        
        # Prepend the structural language tag target
        formatted_input = f"<{tgt_lang}> {text}"
        
        with torch.no_grad():
            # Dynamically reference text token converter
            from pipeline import sentence_to_tensor
            input_tensor = sentence_to_tensor(self.vocab, formatted_input).to(dtype=torch.long, device=self.device)
            
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            
            decoder_outputs, _, _ = self.decoder(
                encoder_hidden=encoder_hidden, 
                encoder_outputs=encoder_outputs, 
                max_len=20,
                teacher_forcing_ratio=0.0
            )
            
            _, topi = decoder_outputs.topk(1)
            decoded_indices = topi.squeeze().tolist()
            
            if isinstance(decoded_indices, int):
                decoded_indices = [decoded_indices]
                
            decoded_words = []
            for idx in decoded_indices:
                if idx == EOS_token:
                    break
                word = self.vocab.index2word.get(idx, "<UNK>")
                if not (word.startswith("<") and word.endswith(">")):
                    decoded_words.append(word)
                
            return ' '.join(decoded_words)

if __name__ == "__main__":
    hidden_size = 64
    pipeline_vocab = MultilingualVocabulary()
    pipeline_vocab.add_sentence("<es> the cat sleeps el gato duerme")
    max_vocab_size = pipeline_vocab.num_words

    encoder = EncoderRNN(max_vocab_size, hidden_size).to(device)
    decoder = AttentionDecoderRNN(hidden_size, max_vocab_size).to(device)

    # FIXED LINE 36: No more nested parentheses/tuples! Inputs are passed independently.
    translator = Translator(encoder, decoder, pipeline_vocab, device)
    
    print("Translator module instantiated inside network.py completely error-free!")
