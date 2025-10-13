"""
CLI Interface Module

Main command-line interface for the Qwen3-VL PDF-to-Text OCR Converter.
"""

import logging
import time
from pathlib import Path
from typing import Optional
import click
from tqdm import tqdm

from .gpu_detector import detect_gpu
from .model_selector import ModelSelector
from .model_manager import ModelManager
from .pdf_scanner import PDFScanner, PDFInfo
from .ocr_engine import OCREngine, PDFPageOCR
from .text_generator import TextGenerator, BatchTextGenerator

logger = logging.getLogger(__name__)


class OCRConverter:
    """Main OCR converter orchestrator."""

    def __init__(
        self,
        folder_path: str,
        model_name: str = "auto",
        confidence: float = 0.85,
        max_retries: int = 3,
        skip_existing: bool = False,
        dry_run: bool = False,
        file_pattern: str = "*.pdf",
        verbose: bool = False,
        no_metadata: bool = False
    ):
        """
        Initialize OCR converter.

        Args:
            folder_path: Root folder to process
            model_name: Model name or 'auto'
            confidence: Confidence threshold
            max_retries: Max retry attempts
            skip_existing: Skip PDFs with existing text files
            dry_run: Show what would be processed without processing
            file_pattern: File pattern for filtering
            verbose: Verbose output
            no_metadata: Exclude metadata from output
        """
        self.folder_path = folder_path
        self.model_name = model_name
        self.confidence = confidence
        self.max_retries = max_retries
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.file_pattern = file_pattern
        self.verbose = verbose
        self.no_metadata = no_metadata

        self.gpu_detector = None
        self.model_manager = None
        self.ocr_engine = None

    def run(self) -> int:
        """
        Run the OCR conversion process.

        Returns:
            Exit code (0 for success, 1 for error)
        """
        click.echo("\nQwen3-VL PDF-to-Text OCR Converter")
        click.echo("=" * 50)
        click.echo()

        # 1. Detect GPU
        click.echo("Detecting GPU...")
        self.gpu_detector = detect_gpu()
        has_gpu, gpu_name, vram_gb, compute_cap = self.gpu_detector.get_gpu_info()

        if has_gpu:
            click.echo(f"✓ GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM, Compute {compute_cap})")
            if self.gpu_detector.supports_fp8():
                click.echo("✓ GPU supports FP8 quantization")
            else:
                click.echo("⚠ GPU does not support FP8 (compute capability < 8.9)")
        else:
            click.echo("✗ No GPU detected. Using CPU mode (will be slower)")

        # 2. Select model
        selector = ModelSelector()
        if self.model_name == "auto":
            self.model_name = selector.select_model(vram_gb, compute_cap)
            click.echo(f"✓ Auto-selected model: {self.model_name}")
        else:
            click.echo(f"✓ Using specified model: {self.model_name}")

        # 3. Scan for PDFs
        click.echo(f"\nScanning folder: {self.folder_path}")
        scanner = PDFScanner(
            self.folder_path,
            skip_existing=self.skip_existing,
            file_pattern=self.file_pattern
        )
        pdf_files = scanner.scan_for_pdfs()

        if not pdf_files:
            click.echo("✗ No PDF files found to process")
            return 1

        click.echo(f"✓ Found {len(pdf_files)} PDF file(s) to process")

        if self.skip_existing:
            click.echo(f"  (Skipping PDFs with existing .txt files)")

        # 4. Dry run mode
        if self.dry_run:
            click.echo("\n[DRY RUN MODE - No files will be processed]")
            click.echo("\nFiles that would be processed:")
            for pdf_path in pdf_files[:10]:
                info = PDFInfo(pdf_path)
                click.echo(f"  - {info}")
            if len(pdf_files) > 10:
                click.echo(f"  ... and {len(pdf_files) - 10} more")
            return 0

        # 5. Load model
        click.echo("\nLoading model...")
        device = self.gpu_detector.get_device()
        self.model_manager = ModelManager(self.model_name, device)

        if not self.model_manager.load_model():
            click.echo("✗ Failed to load model")
            return 1

        click.echo(f"✓ Model loaded successfully")

        # 6. Initialize OCR components
        self.ocr_engine = OCREngine(
            model_manager=self.model_manager,
            confidence_threshold=self.confidence,
            max_retries=self.max_retries
        )

        page_ocr = PDFPageOCR(self.ocr_engine)
        text_generator = TextGenerator(
            include_metadata=not self.no_metadata,
            add_page_markers=True
        )
        batch_generator = BatchTextGenerator(text_generator)

        # 7. Process PDFs
        click.echo(f"\nProcessing {len(pdf_files)} PDF(s):")
        click.echo("-" * 50)

        start_time = time.time()

        with tqdm(total=len(pdf_files), desc="PDFs", unit="pdf") as pbar:
            for pdf_path in pdf_files:
                pbar.set_description(f"Processing: {pdf_path.name[:30]}")

                # Process single PDF
                success = self._process_pdf(
                    pdf_path=pdf_path,
                    scanner=scanner,
                    page_ocr=page_ocr,
                    batch_generator=batch_generator
                )

                pbar.update(1)

        total_time = time.time() - start_time

        # 8. Display summary
        self._display_summary(batch_generator, total_time)

        # 9. Cleanup
        self.model_manager.unload_model()

        return 0

    def _process_pdf(
        self,
        pdf_path: Path,
        scanner: PDFScanner,
        page_ocr: PDFPageOCR,
        batch_generator: BatchTextGenerator
    ) -> bool:
        """Process a single PDF file."""
        try:
            pdf_start_time = time.time()

            # Convert PDF to images
            page_images, temp_dir = scanner.pdf_to_images(pdf_path)

            if not page_images:
                logger.error(f"Failed to convert PDF to images: {pdf_path}")
                return False

            # Process pages
            page_results = page_ocr.process_page_images(
                page_images=page_images,
                verbose=self.verbose
            )

            # Generate text file
            processing_time = time.time() - pdf_start_time
            success = batch_generator.add_result(
                pdf_path=pdf_path,
                page_results=page_results,
                model_name=self.model_name,
                processing_time=processing_time
            )

            # Cleanup temp files
            scanner.cleanup_temp_images(temp_dir)

            return success

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return False

    def _display_summary(self, batch_generator: BatchTextGenerator, total_time: float) -> None:
        """Display processing summary."""
        summary = batch_generator.get_summary()

        click.echo("\n" + "=" * 50)
        click.echo("Summary:")
        click.echo("-" * 50)
        click.echo(f"Total PDFs found: {summary['total_pdfs']}")
        click.echo(f"PDFs processed: {summary['successful']}")
        click.echo(f"PDFs failed: {summary['failed']}")
        click.echo(f"Total pages: {summary['total_pages']}")
        click.echo(f"Average confidence: {summary['avg_confidence'] * 100:.1f}%")
        click.echo(f"Total time: {self._format_time(total_time)}")
        click.echo("=" * 50)

        if summary['successful'] > 0:
            click.echo("\n✓ Processing complete!")
            click.echo("Text files saved in same folders as source PDFs.")
        else:
            click.echo("\n✗ No files processed successfully")

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--model', default='auto', help='Model name or "auto" for auto-detection')
@click.option('--confidence', default=0.85, type=float, help='Confidence threshold (0.0-1.0)')
@click.option('--max-retries', default=3, type=int, help='Maximum retry attempts per page')
@click.option('--skip-existing', is_flag=True, help='Skip PDFs with existing .txt files')
@click.option('--dry-run', is_flag=True, help='Show what would be processed')
@click.option('--file-pattern', default='*.pdf', help='Filter PDFs by name pattern')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
@click.option('--no-metadata', is_flag=True, help='Exclude metadata from output')
def main(
    folder: str,
    model: str,
    confidence: float,
    max_retries: int,
    skip_existing: bool,
    dry_run: bool,
    file_pattern: str,
    verbose: bool,
    no_metadata: bool
):
    """
    Qwen3-VL PDF-to-Text OCR Converter

    Recursively converts all PDFs in FOLDER to text files using Qwen3-VL models.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run converter
    converter = OCRConverter(
        folder_path=folder,
        model_name=model,
        confidence=confidence,
        max_retries=max_retries,
        skip_existing=skip_existing,
        dry_run=dry_run,
        file_pattern=file_pattern,
        verbose=verbose,
        no_metadata=no_metadata
    )

    exit_code = converter.run()
    exit(exit_code)


if __name__ == "__main__":
    main()
