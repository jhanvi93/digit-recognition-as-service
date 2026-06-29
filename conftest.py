import sys
from pathlib import Path

# Make the ``src`` layout importable during tests without an editable install.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
