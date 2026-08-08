import os
from setuptools import setup

# Read custom installation variables with fallback defaults
BUILD_WITH_CUDA = os.environ.get("TRANSLATOR_CUDA", "1")
DEFAULT_VOCAB_SIZE = os.environ.get("TRANSLATOR_MAX_VOCAB", "5000")

print(f" Compiling complex_translator Package Layout...")
print(f"    - Build with CUDA acceleration support: {BUILD_WITH_CUDA}")
print(f"    - Default internal vocabulary ceiling : {DEFAULT_VOCAB_SIZE}")

# Save these configuration variables into an internal config file inside your package source
pkg_config_path = os.path.join("src", "complex_translator", "install_config.py")
os.makedirs(os.path.dirname(pkg_config_path), exist_ok=True)

with open(pkg_config_path, "w", encoding="utf-8") as f:
    f.write(f"CUDA_SUPPORT = {BUILD_WITH_CUDA == '1'}\n")
    f.write(f"MAX_VOCAB_SIZE = {int(DEFAULT_VOCAB_SIZE)}\n")

# Call setup() to trigger the standard pyproject.toml build
setup()
