import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class MultilingualVocabulary:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.num_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if not word:
                continue
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.index2word[self.num_words] = word
                self.num_words += 1

    def __len__(self):
        return self.num_words

def sentence_to_tensor(vocab, sentence):
    indexes = []
    for word in sentence.split(' '):
        if not word:
            continue
        if word in vocab.word2index:
            indexes.append(vocab.word2index[word])
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).unsqueeze(0)

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
        
        # Attention Layers (Bahdanau Attention)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_hidden, encoder_outputs, target_tensor=None, max_len=20, teacher_forcing_ratio=0.5):
        batch_size = encoder_hidden.size(1) if encoder_hidden.dim() > 2 else encoder_hidden.size(0)
        
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=DEVICE).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs_list = []

        steps = target_tensor.size(1) if target_tensor is not None else max_len
        use_teacher_forcing = True if target_tensor is not None and random.random() < teacher_forcing_ratio else False

        # Pad encoder outputs to match self.max_length if they are shorter
        current_seq_len = encoder_outputs.size(1)
        if current_seq_len < self.max_length:
            padding = torch.zeros(batch_size, self.max_length - current_seq_len, self.hidden_size, device=DEVICE)
            encoder_outputs = torch.cat((encoder_outputs, padding), dim=1)
        elif current_seq_len > self.max_length:
            encoder_outputs = encoder_outputs[:, :self.max_length, :]

        # Extract context matrix step by step
        hidden_for_attn = decoder_hidden.squeeze(0) if decoder_hidden.dim() == 3 else decoder_hidden

        for i in range(steps):
            embedded = self.embedding(decoder_input)
            
            # Combine current word state with total sentence history
            attn_weights = torch.softmax(
                self.attn(torch.cat((embedded[:, 0], hidden_for_attn), 1)), dim=1
            )
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            
            output = torch.cat((embedded[:, 0], context[:, 0]), 1)
            output = self.attn_combine(output).unsqueeze(1)
            output = torch.relu(output)
            
            output, decoder_hidden = self.gru(output, decoder_hidden)
            hidden_for_attn = decoder_hidden.squeeze(0) if decoder_hidden.dim() == 3 else decoder_hidden
            
            top_features = self.out(output)
            decoder_outputs_list.append(top_features)

            if use_teacher_forcing:
                decoder_input = target_tensor[:, i].unsqueeze(1).to(dtype=torch.long, device=DEVICE)
            else:
                _, topi = top_features.topk(1)
                decoder_input = topi.squeeze(-1).detach().to(dtype=torch.long, device=DEVICE)

        final_outputs = torch.cat(decoder_outputs_list, dim=1)
        return final_outputs, decoder_hidden, None

def train_multilingual_pipeline(dataset, vocab, max_vocab_size, epochs=1500, hidden_size=64):
    encoder = EncoderRNN(vocab.num_words, hidden_size).to(DEVICE)
    decoder = AttentionDecoderRNN(hidden_size, max_vocab_size).to(DEVICE)

    enc_optimizer = optim.Adam(encoder.parameters(), lr=0.005)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        
        for item in dataset:
            # We explicitly tag the input text with the target language token
            formatted_src = f"<{item['tgt_lang']}> {item['src_text']}"
            formatted_tgt = f"<{item['tgt_lang']}> {item['tgt_text']}"

            input_tensor = sentence_to_tensor(vocab, formatted_src)
            target_tensor = sentence_to_tensor(vocab, formatted_tgt)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            
            decoder_outputs, _, _ = decoder(
                encoder_hidden=encoder_hidden, 
                encoder_outputs=encoder_outputs, 
                target_tensor=target_tensor,
                teacher_forcing_ratio=0.5
            )

            seq_len = target_tensor.size(1)
            sliced_outputs = decoder_outputs[:, :seq_len, :]

            loss = criterion(
                sliced_outputs.reshape(-1, max_vocab_size),
                target_tensor.reshape(-1)
            )

            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            epoch_loss += loss.item()

        if epoch % 300 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] -> Average Loss: {epoch_loss / len(dataset):.4f}")

    return encoder, decoder, vocab, max_vocab_size

def multilingual_translation(encoder, decoder, vocab, sentence, src_lang, tgt_lang, max_vocab_size):
    encoder.eval()
    decoder.eval()
    
    # Prepend target language tag to match training format
    formatted_input = f"<{tgt_lang}> {sentence}"
    
    with torch.no_grad():
        input_tensor = sentence_to_tensor(vocab, formatted_input)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_outputs, _, _ = decoder(
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
            word = vocab.index2word.get(idx, "<UNK>")
            if not (word.startswith("<") and word.endswith(">")):
                decoded_words.append(word)
            
        return ' '.join(decoded_words)

# Define a local wrapper that locks in the models automatically
def translate(sentence, src_lang, tgt_lang):
    return multilingual_translation(
        model_encoder, model_decoder, pipeline_vocab,
        sentence, src_lang, tgt_lang, vocab_limit
    )

if __name__ == "__main__":
    # 1. Path to your CSV file
    csv_file_path = "translate.csv"  # Update this to your actual file path
    
    # Check if the file exists before running
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Could not find your CSV file at: {csv_file_path}")

    # 2. Load the CSV file into our dataset list structure
    dataset = []
    print(f"Loading dataset from {csv_file_path}...")
    
    with open(csv_file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append({
                "src_text": row["src_text"].strip(),
                "src_lang": row["src_lang"].strip(),
                "tgt_text": row["tgt_text"].strip(),
                "tgt_lang": row["tgt_lang"].strip()
            })
            
    print(f"Successfully loaded {len(dataset)} translation pairs!")

    # 3. Dynamically build the vocabulary from your CSV data
    pipeline_vocab = MultilingualVocabulary()
    for item in dataset:
        pipeline_vocab.add_sentence(f"<{item['tgt_lang']}> {item['src_text']}")
        pipeline_vocab.add_sentence(f"<{item['tgt_lang']}> {item['tgt_text']}")

    vocab_limit = pipeline_vocab.num_words
    print(f"Total vocabulary size: {vocab_limit} words.")

    # 4. Train the network using your CSV data
    print("\nStarting Multilingual Attention Network Training...")
    model_encoder, model_decoder, pipeline_vocab, vocab_limit = train_multilingual_pipeline(
        dataset=dataset,
        vocab=pipeline_vocab,
        max_vocab_size=vocab_limit,
        epochs=1000  # Adjust epochs based on how large your CSV dataset is
    )
    
    print("\nTraining complete! Executing test translation...")

    # 5. Run a test translation from your dataset
    test_sentence = "the cat sleeps"
    res = translate(
        sentence=test_sentence,
        src_lang="en",
        tgt_lang="es"
    )

    print(f"\nResult -> Input: '{test_sentence}' | Output: '{res}'")
