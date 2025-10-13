"""
OCR Engine Module

Core OCR engine with Chain-of-Thought reasoning and intelligent retry mechanism.
"""

import logging
import os
from typing import Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image

from .cot_prompts import get_cot_prompt
from .confidence_scorer import ConfidenceScorer
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class OCREngine:
    """Core OCR engine with CoT reasoning and multi-retry mechanism."""

    def __init__(
        self,
        model_manager: ModelManager,
        confidence_threshold: float = 0.85,
        max_retries: int = 3,
        timeout_seconds: int = 60
    ):
        """
        Initialize OCR engine.

        Args:
            model_manager: Loaded ModelManager instance
            confidence_threshold: Confidence threshold for retries
            max_retries: Maximum retry attempts per page
            timeout_seconds: Timeout for each attempt
        """
        self.model_manager = model_manager
        self.confidence_scorer = ConfidenceScorer(confidence_threshold)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def process_image(
        self,
        image_path: str,
        verbose: bool = False
    ) -> Tuple[str, float, int]:
        """
        Process a single image with multi-retry mechanism.

        Args:
            image_path: Path to image file
            verbose: Enable verbose logging

        Returns:
            Tuple of (extracted_text, confidence_score, attempts_used)
        """
        best_text = ""
        best_confidence = 0.0
        attempts_used = 0

        for attempt in range(1, self.max_retries + 1):
            attempts_used = attempt

            # Get appropriate CoT prompt for this attempt
            prompt = get_cot_prompt(attempt)

            # Apply image enhancement for attempt 3
            img_path = image_path
            if attempt == 3:
                img_path = self._enhance_image(image_path)

            if verbose:
                logger.info(f"Attempt {attempt}/{self.max_retries}")

            # Generate text using VLM
            extracted_text = self.model_manager.generate_text(
                image_path=img_path,
                prompt=prompt,
                max_tokens=4096,
                temperature=0.1
            )

            # Clean up enhanced image if created
            if attempt == 3 and img_path != image_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass

            if not extracted_text:
                logger.warning(f"Attempt {attempt} failed - no text extracted")
                continue

            # Calculate confidence
            confidence = self.confidence_scorer.calculate_confidence(extracted_text)

            if verbose:
                logger.info(f"Confidence: {confidence:.2f}")

            # Keep best result
            if confidence > best_confidence:
                best_text = extracted_text
                best_confidence = confidence

            # Check if we can stop early
            if not self.confidence_scorer.needs_retry(confidence):
                if verbose:
                    logger.info(f"✓ High confidence achieved on attempt {attempt}")
                break

            if verbose and attempt < self.max_retries:
                logger.info(f"Low confidence ({confidence:.2f}), retrying...")

        return best_text, best_confidence, attempts_used

    def _enhance_image(self, image_path: str) -> str:
        """
        Enhance image quality for better OCR (used in attempt 3).

        Args:
            image_path: Path to original image

        Returns:
            Path to enhanced image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Sharpen
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # Save enhanced image
            base, ext = os.path.splitext(image_path)
            enhanced_path = f"{base}_enhanced{ext}"
            cv2.imwrite(enhanced_path, sharpened)

            logger.debug(f"Image enhanced: {enhanced_path}")
            return enhanced_path

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}, using original")
            return image_path


class PDFPageOCR:
    """OCR processor for PDF pages."""

    def __init__(self, ocr_engine: OCREngine):
        """
        Initialize PDF page OCR processor.

        Args:
            ocr_engine: OCREngine instance
        """
        self.ocr_engine = ocr_engine

    def process_page_images(
        self,
        page_images: List[str],
        verbose: bool = False
    ) -> List[Tuple[str, float, int]]:
        """
        Process multiple page images.

        Args:
            page_images: List of paths to page images
            verbose: Enable verbose logging

        Returns:
            List of tuples (text, confidence, attempts) for each page
        """
        results = []

        for i, image_path in enumerate(page_images):
            if verbose:
                logger.info(f"Processing page {i + 1}/{len(page_images)}")

            text, confidence, attempts = self.ocr_engine.process_image(
                image_path, verbose=verbose
            )

            results.append((text, confidence, attempts))

            if verbose:
                logger.info(f"Page {i + 1} completed - Confidence: {confidence:.2f}, Attempts: {attempts}")

        return results


def create_ocr_engine(
    model_manager: ModelManager,
    confidence_threshold: float = 0.85,
    max_retries: int = 3
) -> OCREngine:
    """
    Convenience function to create OCR engine.

    Args:
        model_manager: Loaded ModelManager instance
        confidence_threshold: Confidence threshold
        max_retries: Maximum retries per page

    Returns:
        OCREngine instance
    """
    return OCREngine(
        model_manager=model_manager,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries
    )


if __name__ == "__main__":
    # Test OCR engine
    logging.basicConfig(level=logging.INFO)
    print("OCR Engine Test")
    print("=" * 80)
    print("\nNote: This test requires:")
    print("1. A test image file")
    print("2. GPU with sufficient VRAM")
    print("3. Internet connection for model download")
    print("\nSkipping actual inference test in standalone mode.")
