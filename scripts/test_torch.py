import sys

# Test 1: Without clearing
import paddleformers
print("After paddleformers import:")
from transformers.utils.import_utils import is_torch_available
print(f"PyTorch available: {is_torch_available()}")  # False

# Test 2: With clearing
if 'transformers' in sys.modules:
    del sys.modules['transformers']
print("\nAfter clearing transformers:")
from transformers.utils.import_utils import is_torch_available
print(f"PyTorch available: {is_torch_available()}")  # True
