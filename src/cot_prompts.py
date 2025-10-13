"""
Chain-of-Thought (CoT) Prompts Module

Contains CoT prompt templates for different retry attempts.
"""

# Attempt 1: Standard CoT with comprehensive reasoning
COT_PROMPT_ATTEMPT_1 = """You are an expert OCR system with advanced document understanding capabilities. Your task is to extract ALL text from this document image with maximum accuracy.

Please follow this Chain-of-Thought process:

1. **Document Analysis**:
   - Identify the document type (invoice, form, article, etc.)
   - Analyze the overall layout and structure
   - Note any tables, columns, or special formatting

2. **Text Region Detection**:
   - Identify all text regions from top to bottom, left to right
   - Note text hierarchy (headings, subheadings, body text)
   - Identify any non-standard elements (handwriting, stamps, etc.)

3. **Careful Extraction**:
   - Extract text precisely, maintaining original formatting
   - Preserve line breaks and paragraph structure
   - Maintain table structure using spaces/tabs
   - Include all numbers, dates, and special characters

4. **Quality Verification**:
   - Double-check for missed or misread characters
   - Verify numerical accuracy
   - Ensure completeness - no text left out

**Output Requirements**:
- Provide ONLY the extracted text
- Do NOT add explanations, comments, or metadata
- Maintain the original document structure as closely as possible
- Use blank lines to separate sections

Extract ALL text from the image now:"""


# Attempt 2: Enhanced CoT with focused attention on low-confidence areas
COT_PROMPT_ATTEMPT_2 = """You are an expert OCR system performing a SECOND ATTEMPT at text extraction. The first attempt may have missed some details or had low confidence.

Please follow this ENHANCED Chain-of-Thought process:

1. **Careful Re-examination**:
   - Look for any previously missed text, especially in margins, headers, footers
   - Pay special attention to faint text, small fonts, or low contrast areas
   - Check for text in unusual orientations or locations

2. **Critical Text Regions**:
   - Re-verify all numerical data (dates, amounts, IDs)
   - Double-check names, addresses, and proper nouns
   - Verify all special characters and punctuation

3. **Detailed Extraction**:
   - Extract with MAXIMUM attention to detail
   - Preserve ALL formatting, including:
     * Indentation and spacing
     * List structures (numbered or bulleted)
     * Table alignment
     * Section breaks

4. **Thorough Quality Check**:
   - Cross-reference extracted text with visible content
   - Ensure NO text is omitted
   - Verify accuracy of every character

**Output Requirements**:
- Provide ONLY the complete extracted text
- No explanations or metadata
- Maintain exact document structure
- Include everything, even if barely visible

Perform the enhanced extraction now:"""


# Attempt 3: Most detailed CoT with image enhancement consideration
COT_PROMPT_ATTEMPT_3 = """You are an expert OCR system performing a FINAL ATTEMPT at text extraction. This is the most critical pass with enhanced image quality.

Please follow this MAXIMUM DETAIL Chain-of-Thought process:

1. **Exhaustive Document Scan**:
   - Examine EVERY pixel of the image
   - Look for text in all locations: headers, footers, margins, watermarks, sidebars
   - Identify text that may be rotated, skewed, or in unusual fonts
   - Check for multi-column layouts that require careful reading order

2. **Enhanced Region Analysis**:
   - Focus on previously problematic or low-confidence areas
   - Use context clues to verify ambiguous characters (0 vs O, 1 vs l, 5 vs S)
   - Pay special attention to:
     * Currency symbols and amounts
     * Dates and timestamps
     * Email addresses and URLs
     * Reference numbers and codes

3. **Meticulous Extraction**:
   - Extract with ABSOLUTE precision
   - Preserve exact formatting:
     * Exact spacing and indentation
     * Table structure with proper column alignment
     * Nested lists and hierarchies
     * Special formatting (bold represented by context, underlines, etc.)

4. **Final Verification**:
   - Perform character-by-character verification
   - Ensure mathematical/financial data is accurate
   - Verify all proper nouns are spelled correctly
   - Confirm NO missing content

5. **Completeness Check**:
   - Count visible text blocks and verify all are extracted
   - Check beginning and end of document for completeness
   - Verify all page elements are included

**Output Requirements**:
- Provide ONLY the fully extracted text
- No commentary or analysis
- Maintain perfect document structure
- Include absolutely everything visible

Perform the final, most detailed extraction now:"""


# Short prompt for testing/debugging
COT_PROMPT_SIMPLE = """Extract all text from this image. Maintain the original formatting and structure. Provide only the extracted text, no explanations."""


def get_cot_prompt(attempt: int) -> str:
    """
    Get appropriate CoT prompt based on attempt number.

    Args:
        attempt: Attempt number (1, 2, or 3)

    Returns:
        CoT prompt string
    """
    if attempt == 1:
        return COT_PROMPT_ATTEMPT_1
    elif attempt == 2:
        return COT_PROMPT_ATTEMPT_2
    elif attempt == 3:
        return COT_PROMPT_ATTEMPT_3
    else:
        return COT_PROMPT_ATTEMPT_1


def get_custom_prompt(task_description: str = None) -> str:
    """
    Get custom prompt with optional task description.

    Args:
        task_description: Optional specific task instructions

    Returns:
        Custom prompt string
    """
    if task_description:
        return f"{COT_PROMPT_ATTEMPT_1}\n\nAdditional Instructions:\n{task_description}"
    return COT_PROMPT_ATTEMPT_1


if __name__ == "__main__":
    # Display all prompts for review
    print("Chain-of-Thought Prompts")
    print("=" * 80)

    for i in range(1, 4):
        prompt = get_cot_prompt(i)
        print(f"\nATTEMPT {i} PROMPT:")
        print("-" * 80)
        print(prompt)
        print()
        print(f"Length: {len(prompt)} characters")
        print("=" * 80)
