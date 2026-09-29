import sys
import os

# Ensure algebra and arithmetic are in pythonpath for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
algebra_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(algebra_dir)

if algebra_dir not in sys.path:
    sys.path.insert(0, algebra_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
