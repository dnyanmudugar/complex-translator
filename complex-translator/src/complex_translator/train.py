import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import local project modules
from vocabulary import Vocabulary

# 1. Custom Translation Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_numericalized = self.src_vocab.numericalize(self.src_sentences[index])
        trg_numericalized = self.trg_vocab.numericalize(self.trg_sentences[index])
        
        return (
            torch.tensor(src_numericalized, dtype=torch.long),
            torch.tensor(trg_numericalized, dtype=torch.long)
        )

# 2. Collate Function to Pad Sequences in Batches Dynamically
class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Sort batch by source sequence length descending (useful for RNN packing if needed)
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        
        src_tensors = [item[0] for item in batch]
        trg_tensors = [item[1] for item in batch]
        
        # Automatically pad shorter sentences with your PAD token ID
        src_padded = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=self.pad_idx)
        trg_padded = nn.utils.rnn.pad_sequence(trg_tensors, batch_first=True, padding_value=self.pad_idx)
        
        return src_padded, trg_padded

# 3. Main Training Function
def train_model(epochs=10, batch_size=2):
    # Dummy data for demonstration. Replace this with your actual text loading logic!
    raw_src_data = ["hello world", "how are you", "good morning", "see you later"]
    raw_trg_data = ["hola mundo", "como estas", "buenos dias", "hasta luego"]

    # Initialize Vocabularies
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    src_vocab.build_vocab_from_corpus(raw_src_data)
    trg_vocab.build_vocab_from_corpus(raw_trg_data)
    
    # Ensure pad token index matches across both vocabularies
    pad_idx = src_vocab.word2idx[src_vocab.pad_token]

    # Initialize Data Infrastructure
    dataset = TranslationDataset(raw_src_data, raw_trg_data, src_vocab, trg_vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PadCollate(pad_idx=pad_idx)
    )

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_idx, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            
            pass # Remove this line once your model code is uncommented
            
        print(f"Epoch [{epoch+1}/{epochs}] complete.")

if __name__ == "__main__":
    train_model(epochs=10, batch_size=2)
