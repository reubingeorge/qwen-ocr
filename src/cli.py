"""
Command-Line Interface Module

This module provides the CLI for the Qwen3-VL OCR converter.
"""

import sys
import time
from pathlib import Path
import click
from tqdm import tqdm

from .gpu_detector import verify_gpu_requirements
from .model_manager import ModelManager
from .pdf_scanner import PDFScanner
from .ocr_engine import OCREngine
from .text_generator import TextGenerator


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--confidence', default=0.85, type=float,
              help='Confidence threshold for retry (0.0-1.0). Default: 0.85')
@click.option('--max-retries', default=3, type=int,
              help='Maximum retry attempts per page. Default: 3')
@click.option('--skip-existing', is_flag=True,
              help='Skip PDFs that already have .txt files')
@click.option('--dry-run', is_flag=True,
              help='Show what would be processed without actually processing')
@click.option('--file-pattern', default='*.pdf',
              help='Filter PDFs by name pattern. Default: *.pdf')
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose logging')
@click.option('--no-metadata', is_flag=True,
              help='Exclude metadata from text files')
@click.option('--gpu-memory', default=0.70, type=float,
              help='GPU memory utilization (0.1-0.95). Default: 0.70')
def main(folder, confidence, max_retries, skip_existing, dry_run,
         file_pattern, verbose, no_metadata, gpu_memory):
    """
    Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter

    Recursively converts all PDFs in FOLDER to text files using
    state-of-the-art Vision Language Model with Chain-of-Thought reasoning.

    Example:
        python ocr_converter.py /path/to/documents

        python ocr_converter.py /path/to/documents --skip-existing --verbose
    """
    # Print header
    print_header()

    # Step 1: Verify GPU requirements
    print("\n" + "="*60)
    print("STEP 1: Verifying GPU Compatibility")
    print("="*60)

    if not verify_gpu_requirements():
        print("\n❌ GPU verification failed. Exiting.")
        sys.exit(1)

    # Step 2: Scan for PDFs
    print("\n" + "="*60)
    print("STEP 2: Scanning for PDF Files")
    print("="*60)

    try:
        scanner = PDFScanner(folder, skip_existing=skip_existing, file_pattern=file_pattern)
        pdf_files = scanner.scan()
        scanner.print_summary(pdf_files)

        if not pdf_files:
            print("No PDF files to process. Exiting.")
            sys.exit(0)

        if dry_run:
            print("\n=== DRY RUN MODE ===")
            print("The following files would be processed:\n")
            for pdf in pdf_files[:10]:
                print(f"  - {pdf['relative_path']}")
            if len(pdf_files) > 10:
                print(f"  ... and {len(pdf_files) - 10} more")
            print("\nRe-run without --dry-run to process files.")
            sys.exit(0)

    except Exception as e:
        print(f"ERROR: Failed to scan directory: {e}")
        sys.exit(1)

    # Step 3: Load model
    print("\n" + "="*60)
    print("STEP 3: Loading Model")
    print("="*60)

    try:
        model_manager = ModelManager(gpu_memory_utilization=gpu_memory)
        model_manager.load_model()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Step 4: Process PDFs
    print("\n" + "="*60)
    print(f"STEP 4: Processing {len(pdf_files)} PDF Files")
    print("="*60)

    ocr_engine = OCREngine(
        model_manager=model_manager,
        confidence_threshold=confidence,
        max_retries=max_retries,
        verbose=verbose
    )

    text_generator = TextGenerator(include_metadata=not no_metadata)

    # Process statistics
    processed = 0
    failed = 0
    total_pages = 0
    total_time_start = time.time()

    # Process each PDF
    with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
        for pdf_info in pdf_files:
            pdf_path = pdf_info['pdf_path']
            txt_path = pdf_info['txt_path']

            try:
                # Process PDF
                result = ocr_engine.process_pdf(pdf_path)

                if result['success']:
                    # Generate text file
                    if text_generator.generate_text_file(result, txt_path):
                        processed += 1
                        total_pages += result['total_pages']
                    else:
                        failed += 1
                        print(f"  ✗ Failed to create text file for {pdf_info['filename']}")
                else:
                    failed += 1
                    print(f"  ✗ Failed to process {pdf_info['filename']}")

            except Exception as e:
                failed += 1
                print(f"  ✗ Error processing {pdf_info['filename']}: {e}")

            pbar.update(1)

    total_time = time.time() - total_time_start

    # Print summary
    print_summary(processed, failed, total_pages, total_time)

    # Cleanup
    model_manager.unload()


def print_header():
    """Print application header."""
    print("\n" + "="*60)
    print("Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter")
    print("="*60)


def print_summary(processed: int, failed: int, total_pages: int, total_time: float):
    """
    Print processing summary.

    Args:
        processed: Number of successfully processed PDFs
        failed: Number of failed PDFs
        total_pages: Total pages processed
        total_time: Total processing time
    """
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Total pages: {total_pages}")
    print(f"Total time: {format_time(total_time)}")

    if total_pages > 0:
        avg_time_per_page = total_time / total_pages
        print(f"Average time per page: {avg_time_per_page:.1f}s")

    print("="*60)

    if failed == 0:
        print("\n✓ All PDFs converted successfully!")
    else:
        print(f"\n⚠ {failed} PDF(s) failed to convert")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


if __name__ == "__main__":
    main()
