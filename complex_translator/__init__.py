# src/complex_translator/__init__.py

# 1. Explicitly import core functions to make imports cleaner for users
from complex_translator.core import real_translate

# 2. Define package metadata
__version__ = "0.1.0"
__author__ = "dnyanmudugar"

# 3. Define what is exposed when a user runs "from complex_translator import *"
__all__ = ["real_translate"]
