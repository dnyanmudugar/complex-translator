import re
from collections import Counter

class Vocabulary:
    def __init__(self, name: str):
        self.name = name
        # Standard special tokens
        self.PAD_token = 0  # Padding
        self.SOS_token = 1  # Start of Sentence
        self.EOS_token = 2  # End of Sentence
        self.UNK_token = 3  # Unknown words
        
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.num_words = 4

    def add_sentence(self, sentence: str):
        """Cleans and tokenizes a sentence, adding words to the vocabulary."""
        cleaned_words = self.clean_text(sentence)
        for word in cleaned_words:
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.index2word[self.num_words] = word
                self.num_words += 1

    def clean_text(self, text: str) -> List[str]:
        """Lowercases text and separates punctuation symbols."""
        text = str(text).lower().strip()
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.split()

    def encode(self, sentence: str, max_len: int = 15, add_eos: bool = True) -> List[int]:
        """Converts text words into a list of vocabulary index integers."""
        words = self.clean_text(sentence)[:max_len]
        tokens = [self.word2index.get(word, self.UNK_token) for word in words]
        if add_eos and len(tokens) < max_len:
            tokens.append(self.EOS_token)
        # Pad up to maximum sequence target length
        tokens += [self.PAD_token] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, token_ids: List[int]) -> str:
        """Converts vocabulary index integers back into a text string."""
        words = []
        for token_id in token_ids:
            if token_id == self.EOS_token or token_id == self.PAD_token:
                break
            words.append(self.index2word.get(token_id, "<UNK>"))
        return " ".join(words)
