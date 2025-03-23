# ai/data/__init__.py
"""
AI Data Storage
-------------
This package manages the storage and organization of AI training and evaluation data.

Directory Structure:
- raw/: Contains unprocessed data files including:
  - Market data
  - News articles
  - Social media feeds
  - Financial reports

- processed/: Contains cleaned and preprocessed data:
  - Normalized market data
  - Tokenized text
  - Feature vectors
  - Training datasets

Note: The raw/ and processed/ directories are not Python packages themselves
as they only store data files. They should not contain Python modules.
"""

import os
from pathlib import Path

# Define base paths for data directories
DATA_DIR = Path(__file__).parent
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Ensure data directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)