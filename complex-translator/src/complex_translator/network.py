import torch
import torch.nn as nn
import random

# Placeholder setup to test execution within network.py locally
from pipeline import MultilingualVocabulary, EncoderRNN, AttentionDecoderRNN
from device import device

SOS_token = 0
EOS_token = 1

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
