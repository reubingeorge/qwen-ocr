"""
OCR Engine Module

This is the core OCR engine that combines Chain-of-Thought prompting,
confidence scoring, and intelligent retry logic for optimal accuracy.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from .cot_prompts import get_prompt_for_attempt
from .confidence_scorer import ConfidenceScorer
from .model_manager import ModelManager


class OCREngine:
    """Core OCR engine with CoT reasoning and retry logic."""

    def __init__(self, model_manager: ModelManager, confidence_threshold: float = 0.85,
                 max_retries: int = 3, verbose: bool = False):
        """
        Initialize the OCR engine.

        Args:
            model_manager: Loaded model manager instance
            confidence_threshold: Minimum confidence to accept result
            max_retries: Maximum retry attempts per page
            verbose: Enable verbose logging
        """
        self.model_manager = model_manager
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.verbose = verbose
        self.scorer = ConfidenceScorer(threshold=confidence_threshold)

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (higher = better quality, slower)

        Returns:
            List of page images as numpy arrays
        """
        images = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            # Convert each page to image
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Render page to pixmap
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default DPI
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                # Convert RGBA to RGB if needed
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                images.append(img)

            pdf_document.close()

        except Exception as e:
            print(f"ERROR: Failed to convert PDF to images: {e}")
            return []

        return images

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR (used in final retry attempt).

        Args:
            image: Input image as numpy array

        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Apply adaptive thresholding
            enhanced = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # Denoise
            enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)

            # Sharpen
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            # Convert back to RGB
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            return enhanced

        except Exception as e:
            if self.verbose:
                print(f"Warning: Image enhancement failed: {e}")
            return image

    def save_image_temp(self, image: np.ndarray, page_num: int) -> str:
        """
        Save image to temporary file for model processing.

        Args:
            image: Image as numpy array
            page_num: Page number

        Returns:
            Path to temporary image file
        """
        import tempfile

        # Create temp file
        temp_dir = Path(tempfile.gettempdir()) / "qwen_ocr"
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"page_{page_num}_{int(time.time())}.png"

        # Save image
        img_pil = Image.fromarray(image)
        img_pil.save(str(temp_path))

        return str(temp_path)

    def process_page(self, image: np.ndarray, page_num: int) -> Dict[str, any]:
        """
        Process a single page with retry logic.

        Args:
            image: Page image as numpy array
            page_num: Page number

        Returns:
            Dictionary with 'text', 'confidence', 'attempts' keys
        """
        results = []

        for attempt in range(1, self.max_retries + 1):
            if self.verbose:
                print(f"  Attempt {attempt}/{self.max_retries}")

            # Enhance image on final attempt
            current_image = image
            if attempt == self.max_retries and self.max_retries > 1:
                if self.verbose:
                    print("  Applying image enhancement...")
                current_image = self.enhance_image(image)

            # Save image to temp file
            temp_image_path = self.save_image_temp(current_image, page_num)

            try:
                # Determine problem areas from previous attempts
                low_confidence_areas = ""
                if attempt > 1 and results:
                    last_result = results[-1]
                    low_confidence_areas = self.scorer.identify_problem_areas(last_result['text'])

                # Get appropriate prompt
                prompt = get_prompt_for_attempt(attempt, low_confidence_areas)

                # Process with model
                start_time = time.time()
                extracted_text = self.model_manager.process_image_with_prompt(
                    temp_image_path,
                    prompt,
                    max_tokens=4096,
                    temperature=0.1
                )
                processing_time = time.time() - start_time

                # Calculate confidence
                confidence = self.scorer.calculate_confidence(extracted_text)

                result = {
                    'text': extracted_text,
                    'confidence': confidence,
                    'attempt': attempt,
                    'processing_time': processing_time,
                    'enhanced': attempt == self.max_retries and self.max_retries > 1
                }

                results.append(result)

                if self.verbose:
                    print(f"  Confidence: {confidence:.4f} | Time: {processing_time:.1f}s")

                # Check if we can stop early
                if not self.scorer.needs_retry(confidence):
                    if self.verbose:
                        print(f"  ✓ Sufficient confidence achieved")
                    break

                if attempt < self.max_retries:
                    if self.verbose:
                        print(f"  Low confidence, retrying...")

            except Exception as e:
                print(f"  ERROR on attempt {attempt}: {e}")

            finally:
                # Clean up temp file
                try:
                    Path(temp_image_path).unlink()
                except:
                    pass

        # Select best result
        if results:
            best_result = self.scorer.get_best_result(results)
            best_result['total_attempts'] = len(results)
            return best_result
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'attempt': 0,
                'total_attempts': 0,
                'processing_time': 0.0
            }

    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Process an entire PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing: {pdf_path}")

        # Convert PDF to images
        print("  Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)

        if not images:
            print("  ERROR: Failed to convert PDF")
            return {
                'success': False,
                'error': 'Failed to convert PDF to images',
                'pages': []
            }

        print(f"  Pages: {len(images)}")

        # Process each page
        page_results = []
        total_start = time.time()

        for page_num, image in enumerate(images, start=1):
            print(f"  Processing page {page_num}/{len(images)}...")

            page_result = self.process_page(image, page_num)
            page_result['page_number'] = page_num

            page_results.append(page_result)

        total_time = time.time() - total_start

        # Calculate statistics
        avg_confidence = sum(p['confidence'] for p in page_results) / len(page_results) if page_results else 0.0
        total_attempts = sum(p['total_attempts'] for p in page_results)
        pages_with_retry = sum(1 for p in page_results if p['total_attempts'] > 1)

        result = {
            'success': True,
            'pdf_path': pdf_path,
            'pages': page_results,
            'total_pages': len(images),
            'total_time': total_time,
            'avg_confidence': avg_confidence,
            'total_attempts': total_attempts,
            'pages_with_retry': pages_with_retry
        }

        print(f"  ✓ Completed in {total_time:.1f}s")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Pages with retry: {pages_with_retry}/{len(images)}")

        return result


if __name__ == "__main__":
    # Test the OCR engine
    print("OCR Engine Test")
    print("Note: This test requires a loaded model and a PDF file")
