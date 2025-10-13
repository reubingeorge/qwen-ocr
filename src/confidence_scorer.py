"""
Confidence Scoring Module

This module calculates confidence scores for OCR results to determine
if retry is needed and to select the best result among multiple attempts.
"""

import re
from typing import Dict, List


class ConfidenceScorer:
    """Calculates confidence scores for OCR results."""

    def __init__(self, threshold: float = 0.85):
        """
        Initialize the confidence scorer.

        Args:
            threshold: Minimum confidence score (0.0-1.0) to accept result
        """
        self.threshold = threshold

    def calculate_confidence(self, extracted_text: str, image_path: str = None) -> float:
        """
        Calculate a confidence score for extracted text.

        This uses multiple heuristics to estimate quality:
        - Text length and completeness
        - Character variety and distribution
        - Proper word formation
        - Structural elements (spacing, punctuation)
        - Special character handling

        Args:
            extracted_text: The OCR extracted text
            image_path: Optional path to source image (for future enhancements)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not extracted_text or not extracted_text.strip():
            return 0.0

        scores = []

        # 1. Length and completeness score
        scores.append(self._score_length(extracted_text))

        # 2. Character variety score
        scores.append(self._score_character_variety(extracted_text))

        # 3. Word formation score
        scores.append(self._score_word_formation(extracted_text))

        # 4. Structural score
        scores.append(self._score_structure(extracted_text))

        # 5. Special character balance score
        scores.append(self._score_special_characters(extracted_text))

        # 6. Readability score
        scores.append(self._score_readability(extracted_text))

        # Calculate weighted average
        weights = [0.15, 0.20, 0.25, 0.15, 0.10, 0.15]
        confidence = sum(s * w for s, w in zip(scores, weights))

        return round(confidence, 4)

    def _score_length(self, text: str) -> float:
        """
        Score based on text length (very short or very long may indicate issues).

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        length = len(text.strip())

        if length < 10:
            return 0.3  # Too short
        elif length < 50:
            return 0.6  # Short but acceptable
        elif length < 100:
            return 0.8
        elif length < 10000:
            return 1.0  # Good length
        else:
            return 0.9  # Very long, slightly suspicious

    def _score_character_variety(self, text: str) -> float:
        """
        Score based on character variety (good documents have diverse characters).

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        unique_chars = len(set(text.lower()))
        total_chars = len(text)

        if total_chars == 0:
            return 0.0

        # Calculate variety ratio
        variety_ratio = unique_chars / min(total_chars, 100)

        # Good variety is between 0.15 and 0.50
        if 0.15 <= variety_ratio <= 0.50:
            return 1.0
        elif variety_ratio < 0.15:
            return max(0.3, variety_ratio / 0.15)
        else:
            return max(0.5, 1.0 - (variety_ratio - 0.50))

    def _score_word_formation(self, text: str) -> float:
        """
        Score based on proper word formation (words should have vowels, reasonable length).

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)

        if not words:
            return 0.5  # No words found, but might be numbers/tables

        # Count words with vowels
        vowel_pattern = re.compile(r'[aeiouAEIOU]')
        words_with_vowels = sum(1 for word in words if vowel_pattern.search(word))

        if len(words) == 0:
            return 0.5

        vowel_ratio = words_with_vowels / len(words)

        # Most real words have vowels (expect >70%)
        if vowel_ratio >= 0.70:
            return 1.0
        elif vowel_ratio >= 0.50:
            return 0.8
        elif vowel_ratio >= 0.30:
            return 0.6
        else:
            return 0.4

    def _score_structure(self, text: str) -> float:
        """
        Score based on document structure (proper spacing, punctuation, line breaks).

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        scores = []

        # Check for reasonable spacing
        spaces = text.count(' ')
        total_chars = len(text)
        if total_chars > 0:
            space_ratio = spaces / total_chars
            # Good documents have 10-25% spaces
            if 0.10 <= space_ratio <= 0.25:
                scores.append(1.0)
            elif 0.05 <= space_ratio <= 0.35:
                scores.append(0.7)
            else:
                scores.append(0.4)

        # Check for punctuation
        punctuation = len(re.findall(r'[.,;:!?]', text))
        if punctuation > 0:
            scores.append(1.0)
        else:
            scores.append(0.6)  # Tables might not have punctuation

        # Check for line breaks
        lines = text.split('\n')
        if len(lines) > 1:
            scores.append(1.0)
        else:
            scores.append(0.7)

        # Check for reasonable line lengths
        if len(lines) > 1:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            if 10 <= avg_line_length <= 200:
                scores.append(1.0)
            else:
                scores.append(0.7)

        return sum(scores) / len(scores) if scores else 0.5

    def _score_special_characters(self, text: str) -> float:
        """
        Score based on special character balance (too many can indicate OCR errors).

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        # Count special characters (excluding common punctuation and spaces)
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?()\-\'"/@#$%&]', text))
        total_chars = len(text)

        if total_chars == 0:
            return 0.0

        special_ratio = special_chars / total_chars

        # Very few special chars is good
        if special_ratio <= 0.02:
            return 1.0
        elif special_ratio <= 0.05:
            return 0.9
        elif special_ratio <= 0.10:
            return 0.7
        else:
            return 0.5  # Too many special chars might indicate OCR issues

    def _score_readability(self, text: str) -> float:
        """
        Score based on readability indicators.

        Args:
            text: The extracted text

        Returns:
            Score between 0.0 and 1.0
        """
        scores = []

        # Check for excessive repeated characters (OCR errors)
        repeated_pattern = re.compile(r'(.)\1{4,}')
        if not repeated_pattern.search(text):
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Check for proper capitalization (not all caps or all lowercase)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if words:
            caps_words = sum(1 for w in words if w[0].isupper())
            caps_ratio = caps_words / len(words)
            if 0.05 <= caps_ratio <= 0.50:
                scores.append(1.0)
            else:
                scores.append(0.7)

        # Check for digit balance (not all digits)
        digits = len(re.findall(r'\d', text))
        letters = len(re.findall(r'[a-zA-Z]', text))
        total = digits + letters
        if total > 0:
            digit_ratio = digits / total
            if digit_ratio <= 0.50:
                scores.append(1.0)
            else:
                scores.append(0.7)

        return sum(scores) / len(scores) if scores else 0.5

    def needs_retry(self, confidence: float) -> bool:
        """
        Determine if a retry is needed based on confidence score.

        Args:
            confidence: The confidence score

        Returns:
            True if retry is needed, False otherwise
        """
        return confidence < self.threshold

    def get_best_result(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Select the best result from multiple OCR attempts.

        Args:
            results: List of dictionaries with 'text', 'confidence', 'attempt' keys

        Returns:
            The result dictionary with highest confidence
        """
        if not results:
            return None

        return max(results, key=lambda x: x.get('confidence', 0.0))

    def identify_problem_areas(self, text: str) -> str:
        """
        Identify specific problem areas in extracted text for targeted retry.

        Args:
            text: The extracted text

        Returns:
            Description of problem areas
        """
        problems = []

        # Check for unusual character sequences
        if re.search(r'[^\w\s]{5,}', text):
            problems.append("unusual character sequences")

        # Check for excessive spacing
        if re.search(r'\s{10,}', text):
            problems.append("irregular spacing")

        # Check for lack of structure
        if '\n' not in text and len(text) > 100:
            problems.append("lack of line breaks")

        # Check for too few words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) < 5 and len(text) > 50:
            problems.append("few recognizable words")

        if not problems:
            return "overall extraction quality"

        return ", ".join(problems)


if __name__ == "__main__":
    # Test the confidence scorer
    scorer = ConfidenceScorer(threshold=0.85)

    # Test cases
    good_text = """
    Invoice #12345
    Date: January 15, 2024

    Bill To:
    John Smith
    123 Main Street
    City, State 12345

    Description          Quantity    Price
    Widget A                    5    $10.00
    Widget B                    3    $15.00

    Total: $95.00
    """

    poor_text = "|||###@@@   aaaaaaaaaa"

    print("=== Good Text ===")
    conf1 = scorer.calculate_confidence(good_text)
    print(f"Confidence: {conf1:.4f}")
    print(f"Needs retry: {scorer.needs_retry(conf1)}")

    print("\n=== Poor Text ===")
    conf2 = scorer.calculate_confidence(poor_text)
    print(f"Confidence: {conf2:.4f}")
    print(f"Needs retry: {scorer.needs_retry(conf2)}")
    print(f"Problem areas: {scorer.identify_problem_areas(poor_text)}")
