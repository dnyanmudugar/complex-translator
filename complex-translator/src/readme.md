# Complex Translator

A custom Python package for **Neural Machine Translation (NMT)**. This framework provides an end-to-end pipeline to build vocabularies, process text corpora, tokenize sentences, and train Deep Learning translation networks.

## Features

- **Dynamic Vocabulary Builder:** Automatically builds source and target vocabulary maps with custom token frequency thresholds.
- **Special Token Handling:** Native support for operational control sequences (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`).
- **Dynamic Batch Padding:** Built-in PyTorch collation utilities to safely pad mismatched variable-length sequence data.
- **Flexible Pipeline:** Designed to plug seamlessly into sequence-to-sequence (Seq2Seq), Attention, or Transformer architectures.

## Installation

Once published, install the package directly using `pip`:

```bash
pip install complex-translator
```

### Development Setup (Local)
To install the package locally in editable mode for modifications:

```bash
git clone https://github.com
cd complex-translator
pip install -e .
```

## Quick Start

Here is a quick example showing how to build a text vocabulary, numericalize sentences into structural tensors, and decode model outputs back into plain text.

```python
from neural_machine_translation.tokenizer import Vocabulary

# 1. Initialize the vocabulary builder
vocab = Vocabulary()

# 2. Build the mappings from a text corpus
corpus = ["hello world", "good morning python", "neural machine translation"]
vocab.build_vocab_from_corpus(corpus, min_freq=1)

print(f"Total vocabulary size: {vocab.num_words}")

# 3. Numericalize a sentence into sequence IDs
sentence = "hello python"
token_ids = vocab.numericalize(sentence)
print(f"Numericalized IDs: {token_ids}")
# Output will be wrapped automatically with <SOS> and <EOS> tokens

# 4. Decode sequence IDs back into strings
decoded_text = vocab.decode(token_ids)
print(f"Decoded string: {decoded_text}")
```

## Running the Training Loop

To run the internal training sequence prototype directly:

```bash
python -m src.complex_translator.train
```

## Project Directory Structure

```text
complex-translator/
├── pyproject.toml
├── README.md
└── src/
    ├── complex_translator/
    │   ├── train.py
    │   └── test.py
    └── neural_machine_translation/
        ├── __init__.py
        ├── tokenizer.py
        └── vocabulary.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
