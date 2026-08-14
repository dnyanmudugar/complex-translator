import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from complex_translator.device import device
from complex_translator.core import TranslationModel
from complex_translator.tokenizer import Vocabulary

def train_model(epochs: int = 5, batch_size: int = 2, learning_rate: float = 0.001):
    print(f"Starting training pipeline using compute hardware: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "your_dataset.csv")
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.DataFrame({
            "source_text": ["hello computer", "machine translation model"],
            "target_text": ["bonjour ordinateur", "modele de traduction automatique"]
        })
        df.to_csv(dataset_path, index=False)

    # 1. Build Vocabularies dynamically from dataset text fields
    src_vocab = Vocabulary("source")
    tgt_vocab = Vocabulary("target")
    
    for job, row in df.iterrows():
        src_vocab.add_sentence(row["source_text"])
        tgt_vocab.add_sentence(row["target_text"])
        
    print(f"Vocabularies built. Source Vocabulary: {src_vocab.num_words} words. Target Vocabulary: {tgt_vocab.num_words} words.")

    # 2. Instantiate model using exact dictionary sizes
    model = TranslationModel(src_vocab.num_words, tgt_vocab.num_words, hidden_dim=256).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Encode sentences using our real vocabulary mappings
            src_list = [src_vocab.encode(text) for text in batch_df["source_text"].tolist()]
            tgt_list = [tgt_vocab.encode(text) for text in batch_df["target_text"].tolist()]
            
            src_batch = torch.tensor(src_list, dtype=torch.long).to(device)
            tgt_batch = torch.tensor(tgt_list, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch[:, :-1]) 
            
            loss = criterion(
                outputs.view(-1, tgt_vocab.num_words), 
                tgt_batch[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average loss: {epoch_loss / max(1, (len(df) // batch_size)):.4f}")

    # Save weights and vocabularies together for deployment inference
    weights_path = os.path.join(current_dir, "translation_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_word2index': src_vocab.word2index,
        'src_index2word': src_vocab.index2word,
        'tgt_word2index': tgt_vocab.word2index,
        'tgt_index2word': tgt_vocab.index2word
    }, weights_path)
    print(f"Model parameters and Vocabularies successfully saved to: {weights_path}")

if __name__ == "__main__":
    train_model(epochs=10, batch_size=2)
