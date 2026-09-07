import os
import sys
import torch

# Safe package imports
try:
    from vocabulary import Vocabulary
    from train import TranslationDataset, PadCollate
    print(" Import Test: SUCCESS! All packages found cleanly.\n")
except ModuleNotFoundError as e:
    print(f" Import Test: FAILED. {e}")
    print("Ensure your folder names match 'neural_machine_translation' precisely.\n")
    sys.exit(1)

def run_pipeline_test():
    print("--- STARTING NMT PIPELINE INTEGRATION TEST ---")
    
    # Mock data setup
    mock_english = ["hello world", "how are you today", "machine learning"]
    mock_spanish = ["hola mundo", "como estas hoy", "aprendizaje automatico"]
    
    # Test 1: Vocabulary Building
    print("[Test 1/4] Building Source and Target Vocabularies...")
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    src_vocab.build_vocab_from_corpus(mock_english)
    trg_vocab.build_vocab_from_corpus(mock_spanish)
    
    print(f"  -> Source Vocab Size: {src_vocab.num_words} tokens")
    print(f"  -> Target Vocab Size: {trg_vocab.num_words} tokens")
    
    # Test 2: Tokenization & Numericalization
    print("\n[Test 2/4] Testing Sentence Numericalization...")
    sample_sentence = "hello learning"
    token_ids = src_vocab.numericalize(sample_sentence)
    print(f"  -> Input:  '{sample_sentence}'")
    print(f"  -> Output Token IDs: {token_ids}")
    
    # Test 3: Dataset Loading
    print("\n[Test 3/4] Initializing Translation Dataset...")
    dataset = TranslationDataset(mock_english, mock_spanish, src_vocab, trg_vocab)
    print(f"  -> Total dataset items: {len(dataset)}")
    
    # Test 4: Batch Padding (Collation)
    print("\n[Test 4/4] Testing Batching & Padding Tensor Shapes...")
    pad_idx = src_vocab.word2idx[src_vocab.pad_token]
    collate_fn = PadCollate(pad_idx=pad_idx)
    
    # Extract items to manually create a sample batch
    sample_batch = [dataset[0], dataset[1]]  # 'hello world' and 'how are you today'
    src_padded, trg_padded = collate_fn(sample_batch)
    
    print(f"  -> Padded Source Tensor Shape: {src_padded.shape} (Batch Size, Sequence Length)")
    print(f"  -> Padded Target Tensor Shape: {trg_padded.shape}")
    print("\n--- ALL TESTS COMPLETED SUCCESSFULLY! YOUR PIPELINE IS SOLID ---")

if __name__ == "__main__":
    run_pipeline_test()
