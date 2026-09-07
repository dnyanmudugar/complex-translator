# Explicitly import core functions to make imports cleaner for users
from core import real_translate

# Define package metadata
__version__ = "0.1.0"
__author__ = "dnyanmudugar"

# Define what is exposed when a user runs "from complex_translator import *"
__all__ = ["real_translate"]
