import math
import os
import pandas as pd
import json

import torch
import torch.nn as nn
import re
import io

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Decoder
from device import device

# file path imports
file_path = "translation_data.csv"
json_file = "tokenizer.json"

# Load your raw dataset
df = pd.read_csv(file_path)

# Add these methods inside your existing MultilingualTokenizer class:
class MultilingualTokenizer:
    def __init__(self):
        """Initializes the WordPiece tokenizer with special tokens and empty maps."""
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = set()
        
        self.initialize_special_tokens()

    def initialize_special_tokens(self):
        """Seeds maps with static structural translation control tokens."""
        special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.vocab.add(token)

    def build_vocab(self, texts: list, max_vocab_size: int = 5000):
        """Trains a WordPiece subword vocabulary using a raw list of texts."""
        word_counts = defaultdict(int)
        for sentence in texts:
            words = re.findall(r"\w+|[.,!?;]", sentence.lower())
            for word in words:
                word_counts[word] += 1

        # Seed initial vocabulary with single characters and subword characters
        char_counts = defaultdict(int)
        for word, count in word_counts.items():
            char_counts[word[0]] += count
            for char in word[1:]:
                char_counts["##" + char] += count

        # Sort by frequency and add top characters to baseline vocabulary
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars:
            if len(self.vocab) >= max_vocab_size:
                break
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char
                self.vocab.add(char)

        # Iteratively learn long subwords based on frequent word clusters
        for word, count in word_counts.items():
            if len(self.vocab) >= max_vocab_size:
                break
            if word not in self.vocab and len(word) > 1:
                new_id = len(self.token_to_id)
                self.token_to_id[word] = new_id
                self.id_to_token[new_id] = word
                self.vocab.add(word)

        print(f"WordPiece Vocabulary built. Final size: {len(self.token_to_id)} tokens.")

    def tokenize_word(self, word: str) -> list:
        """Applies MaxMatch algorithm to slice words into known subword IDs."""
        output_tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            output_tokens.append(cur_substr)
            start = end
        return output_tokens

    def encode(self, text: str, add_special_tokens: bool = True, target_lang_tag: str = None) -> list:
        """Converts raw input text into an index list of numerical token IDs, supporting language tags."""
        words = re.findall(r"\w+|[.,!?;]", text.lower())
        tokens = []
        for word in words:
            tokens.extend(self.tokenize_word(word))
            
        ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]
        
        # If a language tag is provided (e.g., "[es]"), prepend it to the sequence
        if target_lang_tag:
            # Fall back to UNK if the language tag itself isn't in the vocabulary dictionary
            tag_id = self.token_to_id.get(target_lang_tag, self.token_to_id[self.unk_token])
            ids = [tag_id] + ids
            
        if add_special_tokens:
            ids = [self.token_to_id[self.sos_token]] + ids + [self.token_to_id[self.eos_token]]
            
        return ids

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """Translates index arrays back to user-readable strings."""
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.unk_token, self.sos_token, self.eos_token]:
                continue
            tokens.append(token)
            
        # Reconstruct standard word spaces by stripping '##' subword prefixes
        text = " ".join(tokens).replace(" ##", "")
        text = re.sub(r"\s+([.,!?])", r"\1", text).strip()
        return text

    def save_vocab(self, file_path: str = json_file):
        """Saves vocabulary state to a clean JSON layout."""
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved vocabulary to {file_path}")

    def load_vocab(self, file_path: str = json_file):
        """Loads vocabulary mapping state directly out of a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract the vocabulary from the Hugging Face structure
        if "model" in data and "vocab" in data["model"]:
            vocab_dict = data["model"]["vocab"]
        else:
            vocab_dict = data.get("vocab", data)

        # Handle cases where the vocabulary is a list
        if isinstance(vocab_dict, list):
            self.token_to_id = {}
            for item in vocab_dict:
                if isinstance(item, list) and len(item) >= 2:
                    # Format: [["token", score/id], ...]
                    self.token_to_id[str(item[0])] = len(self.token_to_id)
                elif isinstance(item, str):
                    # Format: ["token1", "token2", ...] where index is the ID
                    self.token_to_id[item] = len(self.token_to_id)
        else:
            # Format: {"token": id}
            self.token_to_id = {}
            for k, v in vocab_dict.items():
                if isinstance(v, dict):
                    # If the file saved sub-dictionaries (like an internal word2index tracker),
                    # extract its values directly, or use a primary integer tracking key.
                    if "word2index" in v:
                        for sub_k, sub_v in v["word2index"].items():
                            self.token_to_id[str(sub_k)] = int(sub_v)
                    elif "id" in v:
                        self.token_to_id[str(k)] = int(v["id"])
                    else:
                        # Fallback configuration key tracking
                        print(f"Warning: Skipping unexpected nested dictionary payload for key '{k}'")
                        continue
                else:
                    # Standard flat format string-to-integer conversion path
                    self.token_to_id[str(k)] = int(v)

            # Rebuild your structural reverse index-to-token map dynamically
            self.id_to_token = {int(v): str(k) for k, v in self.token_to_id.items()}
            self.num_words = len(self.token_to_id)

        # Safely build the reverse lookup map
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.vocab = set(self.token_to_id.keys())
        print(f"Loaded vocabulary size: {len(self.token_to_id)} tokens.")

def translate_text(text, target_tag, model, tokenizer, device, max_len=20):
    model.eval()
    src_ids = torch.tensor([tokenizer.encode(text, target_tag)]).to(device)
    tgt_ids = [tokenizer.vocab["<sos>"]]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids]).to(device)
            output = model(src_ids, tgt_tensor)
            next_token = output[0, -1, :].argmax().item()
            tgt_ids.append(next_token)
            if next_token == tokenizer.vocab["<eos>"]:
                break
                
    return tokenizer.decode(tgt_ids)

class MultilingualCSVDataset(Dataset):
    # Change 'tokenizer' to 'tokenizer_instance' here
    def __init__(self, df, tokenizer_instance, is_train=True, all_texts=None): 
        self.df = df
        self.tokenizer = tokenizer_instance  # Ensure it assigns correctly
        
        # Fit tokenizer ONLY during training initialization to simulate real deployment
        if is_train and all_texts:
            self.tokenizer.fit(all_texts)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row['source_text'])
        tgt_text = str(row['target_text'])
        tag = str(row['lang_token'])
        
        src_ids = self.tokenizer.encode(src_text, target_lang_tag=tag)
        tgt_ids = self.tokenizer.encode(tgt_text)
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape [max_len, d_model] representing positional values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Populate alternating sine and cosine waves
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape to [1, max_len, d_model] for batch_first broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so it automatically migrates to CPU/GPU with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Slice the middle dimension up to the current batch's sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultilingualTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Standard PyTorch Transformer configured for [Batch, Seq_Len, Features] layout
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        # Generates a causal upper-triangular mask matrix to prevent looking at future tokens
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # 1. Cast inputs to long integers to safeguard the embedding layer
        src = src.long()
        tgt = tgt.long()
        
        # 2. Source embedding & positional encoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        
        # 3. Target embedding & positional encoding
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        # 4. Correctly generate the subsequent target mask using the current target length
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # 5. Pass through the full Transformer pipeline including padding blocks
        transformer_out = self.transformer(
            src_emb, tgt_emb, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 6. Project final hidden features to dictionary matrix shape
        output = self.fc_out(transformer_out)
        
        return output

class LiveMemoryCheckpoint:
    def __init__(self):
        self.buffer = io.BytesIO()

    def capture(self, model, optimizer, step):
        """Saves state dictionary directly to RAM."""
        self.buffer.seek(0)
        self.buffer.truncate(0)  # Clear previous checkpoint in memory
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.buffer)

    def restore(self, model, optimizer):
        """Loads state dictionary back out of RAM."""
        self.buffer.seek(0)
        checkpoint = torch.load(self.buffer, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    src_padding_mask = (src_padded == 0)
    tgt_padding_mask = (tgt_padded == 0)
    
    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

if __name__ == "__main__":
    # --- SETUP PIPELINE INFRASTRUCTURE ---
    # Example: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime device: {device}")
    
    # Define tokenizer BEFORE creating the translator
    tokenizer = MultilingualTokenizer()
    tokenizer.load_vocab(json_file)

    # 1. Get your vocabulary size from the tokenizer
    vocab_size = len(tokenizer.vocab)
    
    # 3. Now you can save it safely without the NameError!
    memory_bank = LiveMemoryCheckpoint()
    print("Initial model architecture successfully saved to memory!")

    test_row = df.iloc[0]
    sample_src = test_row['source_text']
    sample_tag = test_row['lang_token']
    sample_expected = test_row['target_text']

# Pass both your active encoder and decoder instances as a tuple
# Example: translator = Translator((encoder, decoder), tokenizer, device)
# translate = translator
