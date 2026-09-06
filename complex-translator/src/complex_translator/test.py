import os
import sys
import pandas as pd
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import components from your local pipeline file
from neural_machine_translation import (
    MultilingualTokenizer,
    MultilingualTransformer,
    MultilingualCSVDataset,
    LiveMemoryCheckpoint,
    translate_text,
    device
)

def local_collate_fn(batch):
    """Safely pads varying length sequences into a uniform batch tensor."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    src_padding_mask = (src_padded == 0)
    tgt_padding_mask = (tgt_padded == 0)
    
    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

def setup_mock_environment():
    """Generates dummy files required to initialize infrastructure tests."""
    print("Building mock test environment...")
    
    # 1. Create a dummy translation CSV
    mock_data = {
        'source_text': ['Hello world', 'How are you?', 'Good morning', 'Thank you'],
        'target_text': ['Hola mundo', '¿Cómo estás?', 'Buenos días', 'Gracias'],
        'lang_token': ['[es]', '[es]', '[es]', '[es]']
    }
    pd.DataFrame(mock_data).to_csv('translation_data.csv', index=False)
    
    # 2. Create a basic starting vocabulary JSON layout
    mock_vocab = {
        "vocab": {
            "[PAD]": 0, "[UNK]": 1, "[SOS]": 2, "[EOS]": 3, "[es]": 4,
            "hello": 5, "world": 6, "how": 7, "are": 8, "you": 9,
            "good": 10, "morning": 11, "thank": 12, "hola": 13, "mundo": 14
        }
    }
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(mock_vocab, f)

def run_pipeline_test():
    """Executes a diagnostic smoke test across all structural components."""
    setup_mock_environment()
    
    print("\n[1/5] Initializing Tokenizer and loading state...")
    tokenizer = MultilingualTokenizer()
    tokenizer.load_vocab('tokenizer.json')
    vocab_size = len(tokenizer.token_to_id)
    
    print("[2/5] Constructing PyTorch Dataset & DataLoader pipelines...")
    df = pd.read_csv('translation_data.csv')
    dataset = MultilingualCSVDataset(df, tokenizer, is_train=False)
    
    # Using the locally defined collate function to bypass import issues
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=local_collate_fn)
    
    print("[3/5] Instantiating Multilingual Transformer Network...")
    model = MultilingualTransformer(vocabulary_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("[4/5] Running single-batch mock training step (Forward & Backward pass)...")
    model.train()
    for src, tgt, src_mask, tgt_mask in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        
        # Shift target for autoregressive sequence alignment
        tgt_input = tgt[:, :-1]
        tgt_output_expected = tgt[:, 1:]
        
        # Forward pass
        output = model(src, tgt_input, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask[:, :-1])
        
        # Calculate loss
        loss = criterion(output.reshape(-1, vocab_size), tgt_output_expected.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"--> Step successful. Initial computed Batch Loss: {loss.item():.4f}")
        break  # Only test one batch step
        
    print("[5/5] Testing RAM checkpoint management & inference generation...")
    memory_bank = LiveMemoryCheckpoint()
    memory_bank.capture(model, optimizer, step=1)
    print("    -> Checkpoint captured successfully inside memory buffer.")
    
    # Run an inference prediction loop test
    test_phrase = "Hello world"
    predicted_translation = translate_text(test_phrase, "[es]", model, tokenizer, device)
    print("\nVerification Test Run Complete!")
    print(f"Input text: '{test_phrase}'")
    print(f"Model output string: '{predicted_translation}'")

if __name__ == "__main__":
    try:
        run_pipeline_test()
    except Exception as e:
        print("\n!!! TEST FAILED WITH AN EXCEPTION !!!")
        import traceback
        traceback.print_exc()
    finally:
        # Safe cleanup block that won't crash if files are already gone
        print("\nCleaning up mock test environment...")
        for file in ['translation_data.csv', 'tokenizer.json']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass
        print("Done.")
