class Vocabulary:
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>", sos_token="<SOS>", eos_token="<EOS>"):
        # 1. Initialize maps
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        
        # 2. Assign special tokens
        self.pad_token = pad_token  # Used to make sentences the same length
        self.unk_token = unk_token  # Used for words not in the vocabulary
        self.sos_token = sos_token  # Start of Sentence token
        self.eos_token = eos_token  # End of Sentence token
        
        # 3. Add special tokens to maps sequentially
        for token in [self.pad_token, self.unk_token, self.sos_token, self.eos_token]:
            self.add_word(token)

    def add_word(self, word):
        """Adds a single word to the vocabulary maps."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[word] = 1
        else:
            self.word_counts[word] += 1

    def add_sentence(self, sentence):
        """Splits a sentence by whitespace and adds all tokens."""
        for word in sentence.strip().split():
            self.add_word(word)

    def build_vocab_from_corpus(self, sentences, min_freq=1):
        """
        Builds the entire vocabulary from a list of sentences,
        filtering out words that appear fewer times than min_freq.
        """
        # First count all frequencies
        for sentence in sentences:
            for word in sentence.strip().split():
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
                    
        # Filter and add words matching min_freq threshold
        for word, count in self.word_counts.items():
            if count >= min_freq:
                self.add_word(word)

    def numericalize(self, sentence):
        """Converts a text string sentence into a list of integer token IDs."""
        numericalized = [self.word2idx[self.sos_token]]  # Add start token
        
        for word in sentence.strip().split():
            # If word is missing, default to the UNK token ID
            numericalized.append(self.word2idx.get(word, self.word2idx[self.unk_token]))
            
        numericalized.append(self.word2idx[self.eos_token])  # Add end token
        return numericalized

    def decode(self, indices):
        """Converts a list of integer IDs back into a human-readable string."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            # Stop decoding if we hit the End of Sentence token
            if word == self.eos_token:
                break
            # Skip PAD and SOS tokens in final output string
            if word not in [self.pad_token, self.sos_token]:
                words.append(word)
                
        return " ".join(words)

    def __len__(self):
        """Allows calling len(vocab) to get total unique tokens."""
        return len(self.word2idx)

    def num_words(self):
        """Returns the total number of unique words in the vocabulary."""
        return len(self.word2idx)
