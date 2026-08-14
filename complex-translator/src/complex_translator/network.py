import torch
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
