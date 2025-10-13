"""
Chain-of-Thought (CoT) Prompt Templates

This module contains carefully crafted CoT prompts that guide the Qwen3-VL-30B-A3B-Thinking-FP8
model through a step-by-step reasoning process for optimal OCR accuracy.
"""


def get_base_prompt() -> str:
    """
    Get the base Chain-of-Thought prompt for standard OCR processing.

    This prompt guides the model through:
    1. Document analysis and layout detection
    2. Text region identification and hierarchy
    3. Careful extraction with formatting preservation
    4. Self-verification against visual layout

    Returns:
        The base CoT prompt template
    """
    return """You are an expert OCR system analyzing a document image. Follow this step-by-step process:

STEP 1 - DOCUMENT ANALYSIS:
- Observe the overall document layout and structure
- Identify the document type (form, invoice, letter, table, multi-column, etc.)
- Note any special formatting (bold, italic, underline, etc.)

STEP 2 - TEXT REGION IDENTIFICATION:
- Identify all text regions from top to bottom, left to right
- Recognize text hierarchy (headers, subheaders, body text, footnotes)
- Detect tables, lists, and structured data

STEP 3 - CAREFUL EXTRACTION:
- Extract all text exactly as it appears
- Preserve spacing, alignment, and formatting
- Maintain table structures with proper column alignment
- Preserve line breaks and paragraph structure

STEP 4 - VERIFICATION:
- Double-check extracted text against the image
- Verify numbers, dates, and special characters
- Ensure no text is missing or duplicated

OUTPUT REQUIREMENTS:
- Provide ONLY the extracted text
- Maintain original formatting and structure
- Use plain text (no markdown unless in the original)
- Preserve tables using spaces for alignment
- Keep paragraph breaks and line spacing

Now extract all text from this document image:"""


def get_retry_prompt(low_confidence_areas: str = "") -> str:
    """
    Get an enhanced CoT prompt for retry attempts.

    This prompt includes focused attention on previously low-confidence areas.

    Args:
        low_confidence_areas: Description of areas that need extra attention

    Returns:
        The enhanced retry CoT prompt
    """
    focus_section = ""
    if low_confidence_areas:
        focus_section = f"""
SPECIAL ATTENTION REQUIRED:
Previous extraction had difficulties with: {low_confidence_areas}
Pay extra careful attention to these areas during extraction.
"""

    return f"""You are an expert OCR system performing a careful re-analysis of a document image.
{focus_section}
Follow this detailed step-by-step process:

STEP 1 - DETAILED DOCUMENT ANALYSIS:
- Carefully examine the complete document layout
- Identify document type and all structural elements
- Note all formatting, including subtle details
- Observe text density and spacing patterns

STEP 2 - COMPREHENSIVE TEXT REGION MAPPING:
- Map ALL text regions systematically from top-left to bottom-right
- Identify text hierarchy at multiple levels
- Detect and map all tables, lists, and structured elements
- Note any overlapping or complex layout regions

STEP 3 - METICULOUS EXTRACTION:
- Extract each text element with extreme care
- Preserve ALL formatting, spacing, and alignment
- Maintain precise table structures with proper column alignment
- Preserve ALL line breaks, paragraph structure, and indentation
- Pay special attention to numbers, dates, and special characters

STEP 4 - THOROUGH VERIFICATION:
- Systematically verify each extracted section against the image
- Check that ALL text is included (headers, body, footnotes, page numbers)
- Verify accuracy of numbers, dates, names, and addresses
- Confirm table structures and alignment are correct
- Ensure no text is missing, duplicated, or misplaced

STEP 5 - QUALITY CHECK:
- Review the entire extraction for completeness
- Check that formatting matches the original
- Verify readability and coherence

OUTPUT REQUIREMENTS:
- Provide ONLY the extracted text
- Maintain EXACT original formatting and structure
- Use plain text (no markdown unless in original)
- Preserve tables with precise space-based alignment
- Keep ALL paragraph breaks and line spacing
- Include ALL content visible in the image

Now carefully extract all text from this document image:"""


def get_final_attempt_prompt() -> str:
    """
    Get the most detailed CoT prompt for final retry attempt.

    This is the most comprehensive prompt, used when previous attempts
    had low confidence. It includes image enhancement guidance.

    Returns:
        The final attempt CoT prompt
    """
    return """You are an expert OCR system performing a final, comprehensive analysis of a document image.
This is a critical extraction that requires maximum accuracy and attention to detail.

STEP 1 - THOROUGH DOCUMENT INSPECTION:
- Examine EVERY pixel of the document carefully
- Identify the complete document structure and layout
- Note ALL formatting elements, no matter how subtle
- Observe text size variations, font styles, and emphasis
- Identify any degraded, faded, or difficult-to-read areas

STEP 2 - SYSTEMATIC REGION ANALYSIS:
- Divide document into logical regions
- Process each region individually from top to bottom
- Identify ALL text elements including:
  * Headers and titles
  * Body text and paragraphs
  * Tables and structured data
  * Lists (bulleted, numbered, or plain)
  * Footnotes and annotations
  * Page numbers and metadata
  * Watermarks or background text

STEP 3 - PRECISION EXTRACTION:
- Extract text with character-level precision
- Preserve EXACT spacing and alignment
- Maintain table structures with precise column alignment
- Preserve ALL formatting (bold, italic, underline if detectable)
- Keep indentation and text flow
- Handle multi-column layouts correctly
- Process rotated or angled text if present
- Extract numbers, dates, and special characters with extra care

STEP 4 - CHARACTER-LEVEL VERIFICATION:
- Verify each word and number against the image
- Check for common OCR mistakes:
  * 0 (zero) vs O (letter O)
  * 1 (one) vs l (lowercase L) vs I (uppercase i)
  * 5 vs S
  * 8 vs B
  * Special characters and punctuation
- Confirm ALL text regions are included
- Verify table structures and alignments

STEP 5 - STRUCTURAL VERIFICATION:
- Confirm document hierarchy is preserved
- Verify paragraph breaks and spacing
- Check table column alignment
- Ensure proper reading order

STEP 6 - COMPLETENESS CHECK:
- Scan entire image to ensure nothing is missed
- Verify headers, footers, and side annotations
- Check for text in margins or corners
- Confirm all pages/sections are included

STEP 7 - FINAL QUALITY ASSURANCE:
- Review entire extraction for accuracy
- Verify formatting matches original
- Confirm readability and coherence
- Check for any duplicates or omissions

OUTPUT REQUIREMENTS:
- Provide ONLY the extracted text (no commentary)
- Maintain EXACT original formatting and structure
- Use plain text (no markdown unless present in original)
- Preserve tables with PRECISE space-based alignment
- Keep ALL paragraph breaks, line spacing, and indentation
- Include EVERY piece of text visible in the image
- Maintain proper reading order
- Preserve special characters exactly as they appear

This is your final attempt. Be thorough, accurate, and complete.
Now extract all text from this document image:"""


def get_table_focused_prompt() -> str:
    """
    Get a CoT prompt specifically optimized for documents with tables.

    Returns:
        The table-focused CoT prompt
    """
    return """You are an expert OCR system specializing in tabular data extraction.

STEP 1 - TABLE IDENTIFICATION:
- Locate all tables in the document
- Identify table boundaries and structure
- Count rows and columns for each table
- Identify header rows and data rows

STEP 2 - TABLE STRUCTURE ANALYSIS:
- Determine column widths and spacing
- Identify merged cells or complex structures
- Note any nested tables
- Recognize table borders and separators

STEP 3 - PRECISE TABLE EXTRACTION:
- Extract table headers first
- Extract each row systematically
- Maintain precise column alignment using spaces
- Preserve cell content exactly as shown
- Handle empty cells appropriately

STEP 4 - NON-TABLE TEXT EXTRACTION:
- Extract all text outside tables
- Maintain proper relationship to tables
- Preserve headers, footers, and annotations

STEP 5 - TABLE VERIFICATION:
- Verify all rows and columns are included
- Check column alignment
- Confirm all cell content is accurate
- Verify table headers are clear

OUTPUT REQUIREMENTS:
- Extract ALL content (tables and regular text)
- Use spaces to align table columns
- Preserve table structure precisely
- Include ALL text from the document

Now extract all text from this document:"""


def get_multi_column_prompt() -> str:
    """
    Get a CoT prompt optimized for multi-column layouts.

    Returns:
        The multi-column focused CoT prompt
    """
    return """You are an expert OCR system specializing in multi-column document layouts.

STEP 1 - COLUMN DETECTION:
- Identify the number of columns
- Locate column boundaries
- Determine reading order (left-to-right, top-to-bottom)

STEP 2 - COLUMN PROCESSING:
- Process each column separately
- Maintain proper reading flow
- Identify where columns break
- Handle text spanning multiple columns

STEP 3 - TEXT EXTRACTION:
- Extract text in proper reading order
- Maintain paragraph structure within columns
- Preserve spacing and formatting
- Handle column breaks appropriately

STEP 4 - INTEGRATION:
- Combine columns in correct reading order
- Preserve document flow
- Maintain formatting and structure

OUTPUT REQUIREMENTS:
- Present text in natural reading order
- Preserve paragraph breaks
- Maintain formatting
- Include ALL content from all columns

Now extract all text from this document:"""


def get_prompt_for_attempt(attempt_number: int, low_confidence_areas: str = "",
                           document_type: str = "general") -> str:
    """
    Get the appropriate CoT prompt based on attempt number and document type.

    Args:
        attempt_number: The attempt number (1, 2, or 3)
        low_confidence_areas: Description of problem areas (for retry attempts)
        document_type: Type of document (general, table, multi-column)

    Returns:
        The appropriate CoT prompt for the attempt
    """
    if attempt_number == 1:
        # First attempt - use base prompt or specialized prompt
        if document_type == "table":
            return get_table_focused_prompt()
        elif document_type == "multi-column":
            return get_multi_column_prompt()
        else:
            return get_base_prompt()

    elif attempt_number == 2:
        # Second attempt - enhanced prompt with focused attention
        return get_retry_prompt(low_confidence_areas)

    else:
        # Third attempt - most detailed prompt
        return get_final_attempt_prompt()


if __name__ == "__main__":
    # Test prompt generation
    print("=== Base Prompt ===")
    print(get_base_prompt())
    print("\n=== Retry Prompt ===")
    print(get_retry_prompt("table structures and numbers"))
    print("\n=== Final Attempt Prompt ===")
    print(get_final_attempt_prompt())
