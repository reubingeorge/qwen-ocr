"""
PDF Scanner Module

This module recursively scans directories for PDF files and manages
the list of files to be processed.
"""

import os
from pathlib import Path
from typing import List, Dict
import fnmatch


class PDFScanner:
    """Scans directories for PDF files and manages processing queue."""

    def __init__(self, root_path: str, skip_existing: bool = False, file_pattern: str = None):
        """
        Initialize the PDF scanner.

        Args:
            root_path: Root directory to scan
            skip_existing: Skip PDFs that already have .txt files
            file_pattern: Optional glob pattern to filter PDF files
        """
        self.root_path = Path(root_path)
        self.skip_existing = skip_existing
        self.file_pattern = file_pattern or "*.pdf"

        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

    def scan(self) -> List[Dict[str, str]]:
        """
        Scan the directory tree for PDF files.

        Returns:
            List of dictionaries containing PDF file information
        """
        pdf_files = []

        print(f"\nScanning directory: {self.root_path}")
        print(f"Pattern: {self.file_pattern}")
        print(f"Skip existing: {self.skip_existing}")

        # Walk directory tree
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)

            for filename in files:
                # Check if file matches pattern
                if fnmatch.fnmatch(filename.lower(), self.file_pattern.lower()):
                    pdf_path = root_path / filename
                    txt_path = pdf_path.with_suffix('.txt')

                    # Check if should skip
                    if self.skip_existing and txt_path.exists():
                        continue

                    pdf_info = {
                        'pdf_path': str(pdf_path),
                        'txt_path': str(txt_path),
                        'filename': filename,
                        'directory': str(root_path),
                        'relative_path': str(pdf_path.relative_to(self.root_path))
                    }

                    pdf_files.append(pdf_info)

        return pdf_files

    def get_summary(self, pdf_files: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Get a summary of the scan results.

        Args:
            pdf_files: List of PDF file information dictionaries

        Returns:
            Summary dictionary
        """
        total_pdfs = len(pdf_files)
        unique_dirs = len(set(pdf['directory'] for pdf in pdf_files))

        # Count existing text files
        existing_txt = 0
        for pdf in pdf_files:
            txt_path = Path(pdf['txt_path'])
            if txt_path.exists():
                existing_txt += 1

        return {
            'total_pdfs': total_pdfs,
            'unique_directories': unique_dirs,
            'existing_txt_files': existing_txt,
            'to_process': total_pdfs - existing_txt if self.skip_existing else total_pdfs
        }

    def print_summary(self, pdf_files: List[Dict[str, str]]):
        """
        Print a summary of scan results.

        Args:
            pdf_files: List of PDF file information dictionaries
        """
        summary = self.get_summary(pdf_files)

        print(f"\n{'='*50}")
        print(f"SCAN SUMMARY")
        print(f"{'='*50}")
        print(f"Total PDFs found: {summary['total_pdfs']}")
        print(f"Unique directories: {summary['unique_directories']}")

        if self.skip_existing:
            print(f"Existing text files: {summary['existing_txt_files']}")
            print(f"PDFs to process: {summary['to_process']}")
        else:
            print(f"PDFs to process: {summary['total_pdfs']}")

        print(f"{'='*50}\n")


def list_pdfs(directory: str, pattern: str = "*.pdf") -> List[str]:
    """
    Simple function to list all PDF files in a directory.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for matching files

    Returns:
        List of PDF file paths
    """
    scanner = PDFScanner(directory, file_pattern=pattern)
    pdf_files = scanner.scan()
    return [pdf['pdf_path'] for pdf in pdf_files]


if __name__ == "__main__":
    # Test the scanner
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_scanner.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    try:
        scanner = PDFScanner(directory, skip_existing=True)
        pdf_files = scanner.scan()
        scanner.print_summary(pdf_files)

        if pdf_files:
            print("Sample files:")
            for pdf in pdf_files[:5]:
                print(f"  - {pdf['relative_path']}")

            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
