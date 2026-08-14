import torch
# 1. Import your custom modules from your repository package
# (Modify these imports based on your actual library file structure)
from pipeline import MultilingualVocabulary, EncoderRNN, AttentionDecoderRNN

def test_pipeline():
    print("Initializing components from complex-translator-library...")
    
    # 2. Mock a small vocabulary dataset
    input_lang = MultilingualVocabulary()
    target_lang = MultilingualVocabulary()
    
    input_lang.add_sentence("hello")
    target_lang.add_sentence("bonjour")
    
    # 3. Instantiate your custom network structures
    hidden_size = 64
    try:
        encoder = EncoderRNN(input_size=input_lang.num_words, hidden_size=hidden_size)
        decoder = AttentionDecoderRNN(hidden_size=hidden_size, output_size=target_lang.num_words)
        print("Success: Custom library components instantiated perfectly.")
        
        # 4. Perform a single forward pass test to verify tensor matching
        test_input = torch.tensor([[0]], dtype=torch.long) # Mock <SOS> token
        encoder_outputs, encoder_hidden = encoder(test_input)
        
        print(f"Encoder Hidden State Shape: {encoder_hidden.shape}")
        print("Success: Forward pass executed without dimension mismatches.")
        
    except Exception as e:
        print(f"Error during testing library: {e}")

if __name__ == "__main__":
    test_pipeline()
