# tests/test_translator.py
import pytest
# Replace with the actual module/class names inside your package
from network import Translator

def test_translation_functionality():
    translator = Translator()
    
    # Example assertion test
    result = translator.translate("Hello", src="en", dest="es")
    assert result == "Hola"