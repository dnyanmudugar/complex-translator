# complex-translator
This python module allows you to translate text to another language using neural network. It does not include module named "googletrans" package. It uses CSV file format as well as model trainings and tokenizer.

# Custom AI Translator

A lightweight, fully offline text translation package for Python built on top of Neural Machine Translation packages. It downloads the required translation models on the first run and operates completely local and offline afterward—no API keys required.

## Features

- **100% Offline**: No network calls or API subscription keys needed after the first run.
- **Privacy First**: Your text never leaves your local machine.
- **Easy Integration**: Clean, object-oriented Python API.

## Installation

You can install this package locally using `pip`:

```bash
# Install the package
pip install complex-translator
```

## Quick Start

Here is how to quickly translate text from English to Spanish using the package:

```python
from complex_translator import translate

src_lang = "en"
tgt_lang = "es"

# Initialize the translator (defaults to English 'en' to Spanish 'es')
translation = translate(text, src_lang, tgt_lang)

# Translate text
text = "Hello, welcome to my custom offline translator!"

print(translation)
# Output: Hola, bienvenido a mi traductor sin conexión personalizado!
```

## Changing Languages

To change languages, pass different language codes during initialization (e.g., English `en` to French `fr`):

```python
src_lang = "en"
tgt_lang = "fr"

translation = translate(text, src_lang, tgt_lang)
```

*Note: The very first time you use a specific language pair, the package will download a small model (~300MB). Subsequent runs will be completely instant.*

## License

This project is licensed under the MIT License.
