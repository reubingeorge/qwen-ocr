#!/usr/bin/env python3
"""
Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter

Main entry point for the CLI application.

Usage:
    python ocr_converter.py /path/to/folder [options]

For help:
    python ocr_converter.py --help
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main

if __name__ == "__main__":
    main()
