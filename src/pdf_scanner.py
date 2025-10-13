"""
PDF Scanner Module

Scans folders recursively for PDF files and converts them to images for processing.
"""

import logging
import os
import tempfile
from typing import List, Tuple
from pathlib import Path
import fnmatch

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFScanner:
    """Scans folders for PDF files and manages PDF-to-image conversion."""

    def __init__(
        self,
        folder_path: str,
        skip_existing: bool = False,
        file_pattern: str = "*.pdf"
    ):
        """
        Initialize PDF scanner.

        Args:
            folder_path: Root folder to scan
            skip_existing: Skip PDFs that already have .txt files
            file_pattern: Pattern for filtering PDF files
        """
        self.folder_path = Path(folder_path)
        self.skip_existing = skip_existing
        self.file_pattern = file_pattern

    def scan_for_pdfs(self) -> List[Path]:
        """
        Recursively scan folder for PDF files.

        Returns:
            List of Path objects for PDF files
        """
        pdf_files = []

        logger.info(f"Scanning folder: {self.folder_path}")

        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if fnmatch.fnmatch(filename.lower(), self.file_pattern.lower()):
                    pdf_path = Path(root) / filename

                    # Check if should skip
                    if self.skip_existing:
                        txt_path = pdf_path.with_suffix('.txt')
                        if txt_path.exists():
                            logger.debug(f"Skipping (text file exists): {pdf_path}")
                            continue

                    pdf_files.append(pdf_path)

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return sorted(pdf_files)

    def pdf_to_images(
        self,
        pdf_path: Path,
        dpi: int = 300
    ) -> Tuple[List[str], str]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image rendering

        Returns:
            Tuple of (list of image paths, temp directory path)
        """
        image_paths = []
        temp_dir = tempfile.mkdtemp(prefix="qwen_ocr_")

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            logger.info(f"Converting PDF to images: {pdf_path.name} ({len(doc)} pages)")

            # Convert each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Render page to pixmap
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale matrix for DPI
                pix = page.get_pixmap(matrix=mat)

                # Save as PNG
                image_filename = f"page_{page_num + 1:04d}.png"
                image_path = os.path.join(temp_dir, image_filename)

                pix.save(image_path)
                image_paths.append(image_path)

                logger.debug(f"Rendered page {page_num + 1}/{len(doc)}")

            doc.close()
            logger.info(f"✓ Converted {len(image_paths)} pages to images")

            return image_paths, temp_dir

        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return [], temp_dir

    def cleanup_temp_images(self, temp_dir: str) -> None:
        """
        Clean up temporary image files.

        Args:
            temp_dir: Temporary directory to clean up
        """
        try:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass

                os.rmdir(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")

        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")


class PDFInfo:
    """Information about a PDF file."""

    def __init__(self, pdf_path: Path):
        """
        Initialize PDF info.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.page_count = 0
        self.file_size_mb = 0.0
        self._load_info()

    def _load_info(self) -> None:
        """Load PDF information."""
        try:
            # Get file size
            self.file_size_mb = self.pdf_path.stat().st_size / (1024 * 1024)

            # Get page count
            doc = fitz.open(self.pdf_path)
            self.page_count = len(doc)
            doc.close()

        except Exception as e:
            logger.warning(f"Error loading PDF info: {e}")

    def __str__(self) -> str:
        """String representation."""
        return f"{self.pdf_path.name} ({self.page_count} pages, {self.file_size_mb:.2f} MB)"


def scan_folder(
    folder_path: str,
    skip_existing: bool = False,
    file_pattern: str = "*.pdf"
) -> List[Path]:
    """
    Convenience function to scan folder for PDFs.

    Args:
        folder_path: Root folder path
        skip_existing: Skip PDFs with existing text files
        file_pattern: File pattern filter

    Returns:
        List of PDF paths
    """
    scanner = PDFScanner(folder_path, skip_existing, file_pattern)
    return scanner.scan_for_pdfs()


if __name__ == "__main__":
    # Test PDF scanner
    logging.basicConfig(level=logging.INFO)

    print("PDF Scanner Test")
    print("=" * 80)

    # Test with current directory
    test_dir = "."

    scanner = PDFScanner(test_dir, skip_existing=False)

    print(f"\nScanning: {test_dir}")
    pdfs = scanner.scan_for_pdfs()

    if pdfs:
        print(f"\nFound {len(pdfs)} PDF(s):")
        for pdf in pdfs[:5]:  # Show first 5
            info = PDFInfo(pdf)
            print(f"  - {info}")

        if len(pdfs) > 5:
            print(f"  ... and {len(pdfs) - 5} more")
    else:
        print("\nNo PDF files found")

    print("\nNote: To test PDF-to-image conversion, provide a specific PDF path.")
