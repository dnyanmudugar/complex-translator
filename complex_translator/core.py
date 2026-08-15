import os
import sys
import pandas as pd
import torch
import torch.nn as nn

from complex_translator.device import device  
from complex_translator.layers import EncoderRNN, AttnDecoderRNN

class TranslationModel(nn.Module):
    """
    Main structural network managing the full Encoder-Decoder attention sequence.
    Processes the target sequence step-by-step to prevent structural size mismatches.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = EncoderRNN(src_vocab_size, hidden_dim).to(device)
        self.decoder = AttnDecoderRNN(hidden_dim, tgt_vocab_size).to(device)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src_seq, tgt_seq):
        batch_size = src_seq.size(0)
        tgt_len = tgt_seq.size(1)
        
        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)
        
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        decoder_hidden = encoder_hidden 
        
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(device)
        
        for t in range(tgt_len):
            decoder_input = tgt_seq[:, t].unsqueeze(1)
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[:, t] = decoder_output
            
        return outputs

def real_translate(text: str) -> str:
    """
    Loads saved model weights along with its native vocabulary configurations,
    tokenizes input, and returns a real decoded translation string.
    """
    if not text.strip():
        return ""
        
    max_len = 15
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "translation_model.pth")
    
    if not os.path.exists(weights_path):
        return "[Error]: Trained weights and vocabularies file not found. Please run train.py first."
        
    # 1. Load your consolidated parameters and vocabulary dictionaries
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    src_word2index = checkpoint['src_word2index']
    tgt_index2word = checkpoint['tgt_index2word']
    
    src_vocab_size = len(src_word2index)
    tgt_vocab_size = len(tgt_index2word)
    
    # 2. Reconstruct the exact model layer dimensions
    model = TranslationModel(src_vocab_size, tgt_vocab_size, hidden_dim=256).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Clean and encode input string matching the training tokenizer rules
    cleaned_words = str(text).lower().strip().split()[:max_len]
    tokens = [src_word2index.get(word, 3) for word in cleaned_words] # 3 is UNK_token
    tokens += [0] * (max_len - len(tokens)) # 0 is PAD_token
    
    src_tensor = torch.tensor([tokens[:max_len]], dtype=torch.long).to(device)
    
    # 4. Inference execution loop
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
        decoder_hidden = encoder_hidden
        
        # Start decoding with the <SOS> token (index 1)
        decoder_input = torch.tensor([[1]], dtype=torch.long).to(device)
        translated_words = []
        
        for _ in range(max_len):
            decoder_output, decoder_hidden, _ = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            next_token = decoder_output.argmax(dim=1)
            token_id = next_token.item()
            
            # Stop if the model predicts <EOS> (2) or <PAD> (0)
            if token_id == 2 or token_id == 0:
                break
                
            word = tgt_index2word.get(token_id, "<UNK>")
            translated_words.append(word)
            
            decoder_input = next_token.unsqueeze(1)
            
    return " ".join(translated_words)

if __name__ == "__main__":
    print(f"Core module initialized using compute device: {device}")
    # Try translating a phrase present in your training dataset data
    test_output = real_translate("hello computer")
    print(f"Translated output text: {test_output}")
