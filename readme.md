# Complex Translator

A custom Python package for **Neural Machine Translation (NMT)**. This framework provides an end-to-end pipeline to build vocabularies, process text corpora, tokenize sentences, and train Deep Learning translation networks using PyTorch.

## Features

- **Dynamic Vocabulary Builder:** Automatically builds source and target vocabulary maps with custom token frequency thresholds.
- **Special Token Handling:** Native support for operational control sequences (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`).
- **Dynamic Batch Padding:** Built-in PyTorch collation utilities (`PadCollate`) to safely pad mismatched variable-length sequence data.
- **Integrated Validation:** Production-tested local path resolution architecture to prevent environment import failures.

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

## Installation

### Local Development Setup
To link and install this package locally in editable mode for modifications:

```bash
cd "complex-translator"
pip install -e .
```

### Production Pip Installation
Once published to PyPI, install the package directly using `pip`:

```bash
pip install complex-translator
```

## Testing the Pipeline

To run the built-in system verification suite and validate package components, use the module execution flag from your project root directory:

```bash
python -m src.complex_translator.test
```

## Quick Start Code Example

Here is how to use the underlying modular classes inside your own code files:

```python
from complex_translator import Vocabulary

# 1. Initialize the vocabulary builder
vocab = Vocabulary()

# 2. Build mappings from a text corpus
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
