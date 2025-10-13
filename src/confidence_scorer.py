"""
Confidence Scorer Module

Calculates confidence scores for OCR output to determine if retry is needed.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculates confidence scores for OCR results."""

    def __init__(self, threshold: float = 0.85):
        """
        Initialize confidence scorer.

        Args:
            threshold: Confidence threshold for determining quality (0.0-1.0)
        """
        self.threshold = threshold

    def calculate_confidence(self, extracted_text: str) -> float:
        """
        Calculate confidence score for extracted text.

        Uses multiple heuristics:
        - Text length (longer is better, but not too short)
        - Character variety (not just repeated characters)
        - Presence of common words and patterns
        - Proper formatting (line breaks, punctuation)
        - Absence of common OCR errors

        Args:
            extracted_text: The OCR extracted text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not extracted_text or len(extracted_text.strip()) == 0:
            return 0.0

        scores = []

        # 1. Length score (reasonable length indicates successful extraction)
        length_score = self._score_length(extracted_text)
        scores.append(length_score)

        # 2. Character variety score
        variety_score = self._score_character_variety(extracted_text)
        scores.append(variety_score)

        # 3. Word formation score (real words vs gibberish)
        word_score = self._score_word_formation(extracted_text)
        scores.append(word_score)

        # 4. Formatting score (proper structure)
        format_score = self._score_formatting(extracted_text)
        scores.append(format_score)

        # 5. Error pattern score (absence of common OCR errors)
        error_score = self._score_error_patterns(extracted_text)
        scores.append(error_score)

        # Calculate weighted average
        weights = [0.15, 0.20, 0.25, 0.20, 0.20]
        confidence = sum(s * w for s, w in zip(scores, weights))

        logger.debug(f"Confidence breakdown - Length: {length_score:.2f}, Variety: {variety_score:.2f}, "
                    f"Words: {word_score:.2f}, Format: {format_score:.2f}, Errors: {error_score:.2f}")
        logger.info(f"Overall confidence: {confidence:.2f}")

        return confidence

    def _score_length(self, text: str) -> float:
        """Score based on text length."""
        length = len(text.strip())

        if length == 0:
            return 0.0
        elif length < 50:
            return 0.3  # Very short - might be incomplete
        elif length < 200:
            return 0.7  # Short but reasonable
        elif length < 2000:
            return 1.0  # Good length
        else:
            return 0.95  # Very long - likely complete

    def _score_character_variety(self, text: str) -> float:
        """Score based on character variety (not repetitive)."""
        if len(text) == 0:
            return 0.0

        unique_chars = len(set(text))
        total_chars = len(text)

        # Calculate variety ratio
        variety_ratio = unique_chars / max(total_chars, 1)

        # Good variety: 0.2-0.8 range
        if variety_ratio < 0.05:
            return 0.2  # Very repetitive
        elif variety_ratio < 0.15:
            return 0.5  # Somewhat repetitive
        elif variety_ratio < 0.40:
            return 1.0  # Good variety
        else:
            return 0.8  # Too much variety might indicate noise

    def _score_word_formation(self, text: str) -> float:
        """Score based on presence of real word patterns."""
        words = text.split()

        if len(words) == 0:
            return 0.0

        # Count words with common patterns
        valid_word_count = 0
        for word in words:
            # Check for reasonable word patterns
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 2:
                # Has vowels and consonants
                has_vowel = bool(re.search(r'[aeiouAEIOU]', clean_word))
                has_consonant = bool(re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', clean_word))

                if has_vowel and has_consonant:
                    valid_word_count += 1

        valid_ratio = valid_word_count / len(words)

        if valid_ratio > 0.7:
            return 1.0
        elif valid_ratio > 0.5:
            return 0.8
        elif valid_ratio > 0.3:
            return 0.6
        else:
            return 0.3

    def _score_formatting(self, text: str) -> float:
        """Score based on proper formatting indicators."""
        score = 0.0

        # Check for line breaks (structured text)
        if '\n' in text:
            score += 0.3

        # Check for punctuation (proper sentences)
        punctuation_count = len(re.findall(r'[.!?,;:]', text))
        if punctuation_count > 0:
            score += 0.3

        # Check for capital letters (proper nouns, sentences)
        if re.search(r'[A-Z]', text):
            score += 0.2

        # Check for numbers (often important in documents)
        if re.search(r'\d', text):
            score += 0.2

        return min(score, 1.0)

    def _score_error_patterns(self, text: str) -> float:
        """Score based on absence of common OCR error patterns."""
        score = 1.0

        # Common OCR error patterns
        error_patterns = [
            r'[Il1]{5,}',  # Too many I/l/1 in a row
            r'[O0]{5,}',   # Too many O/0 in a row
            r'\.{4,}',     # Too many dots
            r'\s{5,}',     # Too many spaces
            r'[^\w\s]{10,}',  # Too many special characters in a row
        ]

        for pattern in error_patterns:
            if re.search(pattern, text):
                score -= 0.15

        # Check for extremely short lines (might indicate fragmentation)
        lines = text.split('\n')
        very_short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 3)
        if very_short_lines > len(lines) * 0.5:
            score -= 0.2

        return max(score, 0.0)

    def needs_retry(self, confidence: float) -> bool:
        """
        Determine if extraction needs retry based on confidence.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            True if retry is needed, False otherwise
        """
        return confidence < self.threshold

    def set_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")


def calculate_confidence(text: str, threshold: float = 0.85) -> Tuple[float, bool]:
    """
    Convenience function to calculate confidence and retry decision.

    Args:
        text: Extracted text
        threshold: Confidence threshold

    Returns:
        Tuple of (confidence_score, needs_retry)
    """
    scorer = ConfidenceScorer(threshold)
    confidence = scorer.calculate_confidence(text)
    needs_retry = scorer.needs_retry(confidence)
    return confidence, needs_retry


if __name__ == "__main__":
    # Test confidence scoring
    logging.basicConfig(level=logging.DEBUG)

    scorer = ConfidenceScorer(threshold=0.85)

    test_cases = [
        ("", "Empty text"),
        ("a", "Single character"),
        ("aaaaaaaaaaa", "Repetitive text"),
        ("Hello world! This is a test document with proper formatting.\nIt has multiple lines.\nAnd proper punctuation.",
         "Good quality text"),
        ("Th1s 1s a d0cum3nt w1th s0m3 0CR 3rr0rs", "Text with OCR errors"),
        ("Invoice #12345\nDate: 2024-01-15\nAmount: $150.00\nThank you for your business.", "Invoice example"),
        ("111111 lllll IIIII 00000", "Poor quality text"),
    ]

    print("Confidence Scoring Tests")
    print("=" * 80)

    for text, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Text: {repr(text[:50])}...")
        confidence = scorer.calculate_confidence(text)
        needs_retry = scorer.needs_retry(confidence)
        print(f"Confidence: {confidence:.2f}")
        print(f"Needs retry: {needs_retry}")
        print("-" * 80)
