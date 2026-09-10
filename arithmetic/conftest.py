import sys
import os

# Add parent directory to sys.path so arithmetic can be imported as a top-level package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
