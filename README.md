# Qwen2.5-VL OCR with Confidence Scoring

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![Transformers](https://img.shields.io/badge/Transformers-latest-orange)
![VRAM](https://img.shields.io/badge/VRAM-24GB-yellow)
![License](https://img.shields.io/badge/license-Model%20Dependent-lightgrey)

A professional PDF-to-text OCR solution powered by the Qwen2.5-VL-7B-Instruct vision-language model. This implementation features intelligent retry logic, confidence scoring, and Chain-of-Thought prompting for high-accuracy text extraction from PDF documents.

## Overview

This project provides a robust OCR pipeline that converts PDF documents to text using state-of-the-art vision-language models. The system employs multiple inference attempts per page with temperature variation, confidence scoring to assess output quality, and automatic batch processing capabilities.

## Key Features

### Vision-Language Model Architecture
Utilizes Qwen2.5-VL-7B-Instruct with AutoModelForVision2Seq, specifically designed for vision-to-sequence tasks such as image captioning and optical character recognition.

### Chain-of-Thought Prompting
Guides the model through systematic reasoning steps for improved accuracy:
1. Document structure identification
2. Sequential text extraction
3. Formatting preservation
4. Verification of numbers and proper nouns
5. Complete text transcription

### Intelligent Retry Mechanism
Multiple OCR attempts per page with temperature scheduling (0.1, 0.2, 0.3) to generate diverse outputs. The best result is selected using medical-grade quality scoring that prioritizes accuracy over completeness.

### Medical-Grade Quality Gating
Optimized for medical pathology reports where accuracy is critical:
- Perplexity-based quality scoring (80% weight on accuracy, 20% on completeness)
- Automatic quality warnings for low-confidence outputs
- Detection of uncertain tokens and inconsistent attempts
- Conservative temperature bias to prevent hallucinations

### Batch Processing
Recursive directory traversal to process entire folder hierarchies with automatic skip of previously processed files.

### Memory-Optimized Inference
BFloat16 precision reduces memory usage by approximately 50% while maintaining numerical stability, enabling efficient processing on consumer GPUs.

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Minimum 24GB VRAM required
- Sufficient disk space for model weights (approximately 15GB)

### Software
- Python 3.8 or higher
- CUDA 11.8 or higher
- NVIDIA drivers with GPU support

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd qwen-ocr
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install qwen-vl-utils pillow requests
pip install pdf2image pymupdf pillow
```

### Step 4: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## Usage

### Basic OCR Example

The system is designed to be used through a Jupyter notebook interface. The main workflow involves:

1. **Load the model** (one-time operation per session)
2. **Process individual PDFs** or **batch process directories**

### Processing a Single PDF

```python
from ocr_functions import ocr_pdf, save_results

# Process PDF with default settings
results = ocr_pdf(
    pdf_path="path/to/document.pdf",
    dpi=300,
    attempts_per_page=3,
    use_cot=True
)

# Save results to text file
save_results(
    results=results,
    filename="output.txt",
    include_metadata=True
)
```

### Batch Processing Directory

```python
from ocr_functions import process_pdf_batch

# Process all PDFs in directory tree
results = process_pdf_batch(
    root_directory='documents',
    dpi=300,
    attempts_per_page=3,
    skip_processed=True
)
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pdf_path` | Path to PDF file | Required |
| `dpi` | Rendering resolution (72-600) | 300 |
| `attempts_per_page` | OCR retry attempts per page | 3 |
| `use_cot` | Enable Chain-of-Thought prompting | True |
| `max_pages` | Limit pages to process | None |
| `skip_processed` | Skip files with existing output | True |
| `max_new_tokens` | Maximum tokens per generation | 2048 |

## Architecture Details

### PDF Processing Pipeline

1. **PDF to Images Conversion**: PyMuPDF renders each page at specified DPI
2. **Image Preprocessing**: Automatic resizing of oversized images (max 2048px)
3. **Vision-Language Processing**: Model processes image and prompt together
4. **Token Generation**: Text generated with configurable parameters
5. **Confidence Calculation**: Multiple metrics computed from generation scores
6. **Medical-Grade Selection**: Best output selected using composite quality score
7. **Quality Gating**: Automatic flagging of pages requiring manual review

### Model Specifications

- **Model Name**: Qwen/Qwen2.5-VL-7B-Instruct
- **Architecture**: AutoModelForVision2Seq
- **Precision**: BFloat16 (torch.bfloat16)
- **Device Mapping**: Automatic distribution across GPU/CPU
- **Memory Footprint**: Approximately 20-24GB VRAM

### Generation Parameters

- **Temperature Schedule**: [0.1, 0.2, 0.3] across attempts
- **Top-p Sampling**: 0.95 threshold
- **Repetition Penalty**: 1.1
- **Max Tokens**: 2048 per page

## Medical-Grade OCR Selection Strategy

This implementation is optimized for medical pathology reports where accuracy is paramount. The selection strategy heavily weights quality over completeness to prevent hallucinations or misreads that could impact clinical decision-making.

### Composite Quality Score

The system calculates a composite score for each OCR attempt combining multiple factors:

**Quality Component (80% weight)**:
- Perplexity-based scoring converted to 0-1 scale
- Lower perplexity indicates higher model confidence
- Typical range: 1.0 (perfect) to 100+ (very uncertain)

**Completeness Component (20% weight)**:
- Normalized character count relative to expected page length
- Ensures output is reasonably complete while prioritizing accuracy

**Penalty Factors**:
- Minimum token probability penalty (30% reduction if min_prob < 0.05)
- Temperature bias (5% penalty for temperatures > 0.1)
- Favors conservative, high-confidence outputs

### Quality Gating Thresholds

Pages are automatically flagged for manual review when:
- Composite quality score < 0.5
- Perplexity > 20.0 (high uncertainty)
- Minimum token probability < 0.05 (very uncertain tokens detected)
- Coefficient of variation > 30% across attempts (inconsistent results)

### Confidence Metrics

Each OCR output includes comprehensive confidence metrics:

**Mean Probability**: Average probability across all generated tokens (0-1 scale)

**Mean Log Probability**: Logarithmic average for numerical stability with very small probabilities

**Perplexity**: Exponential of negative mean log probability measuring model uncertainty

**Minimum Probability**: Identifies the least confident token for spotting potential errors

## Performance Characteristics

### Processing Speed
- Best case: 5-10 seconds per page
- Average: 10-20 seconds per page
- Worst case (3 retries): 25-40 seconds per page

### Memory Usage
- Model loading: 20-22GB VRAM
- During inference: 22-24GB VRAM
- Peak usage with large images: Up to 24GB VRAM

### Accuracy Factors
- DPI setting (300 recommended for standard documents)
- Image quality and resolution
- Font size and clarity
- Document complexity (tables, multi-column layouts)
- Temperature scheduling effectiveness

## Output Format

### Text File Structure

```
================================================================================
PAGE 1
================================================================================
[Transcribed text from page 1]

================================================================================
PAGE 2
================================================================================
[Transcribed text from page 2]
```

### Metadata (Optional)

When `include_metadata=True`, output includes:
- Processing timestamp
- Total pages processed
- Total characters extracted
- Average confidence score
- Total processing time
- Per-page statistics

### Detailed JSON Output

For comprehensive analysis, detailed JSON output includes:
- All retry attempts per page
- Individual confidence scores
- Temperature used for each attempt
- Processing time per attempt
- Character and word counts

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: RuntimeError during model loading or inference

**Solutions**:
```python
# Reduce image size
preprocess_image_for_ocr(image, max_size=1024)

# Clear GPU cache between operations
torch.cuda.empty_cache()

# Reduce max_new_tokens
ocr_pdf(pdf_path, max_new_tokens=1024)
```

### Low Confidence Scores

**Symptoms**: Perplexity > 50 or mean_probability < 0.3

**Solutions**:
- Increase DPI for better image quality (try 400-600)
- Verify image preprocessing isn't over-compressing
- Check source document quality
- Review and adjust Chain-of-Thought prompt

### Slow Processing

**Symptoms**: Processing time exceeds expected ranges

**Solutions**:
```bash
# Monitor GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

# Check for thermal throttling
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Reduce retry attempts for testing
ocr_pdf(pdf_path, attempts_per_page=1)
```

### Import Errors

**Symptoms**: ModuleNotFoundError for transformers or qwen_vl_utils

**Solutions**:
```bash
# Reinstall core dependencies
pip install --upgrade transformers accelerate
pip install qwen-vl-utils --no-cache-dir

# Verify installations
python -c "import transformers; print(transformers.__version__)"
```

## Project Structure

```
qwen-ocr/
├── qwen_ocr_notebook.ipynb    # Main implementation notebook
├── README.md                   # This documentation
├── reports/                    # Example output directory
└── documents/                  # Example input directory
```

## Implementation Notes

### Why Medical-Grade Quality Gating?
For pathology reports, a single misread value (tumor size, staging, biomarker status) can impact treatment decisions. The 80/20 quality-to-completeness weighting ensures accuracy takes precedence over extracting every word. Better to flag a page for manual review than risk introducing hallucinated data into medical records.

### Why Perplexity Over Mean Probability?
Perplexity is more sensitive to outlier tokens and provides better theoretical grounding for uncertainty quantification. A single very low-probability token (potential hallucination) affects perplexity more than mean probability, making it superior for detecting problematic outputs in critical applications.

### Why PyMuPDF (fitz)?
PyMuPDF provides faster rendering and better memory efficiency compared to alternatives like pdf2image, and doesn't require external dependencies like Poppler.

### Why BFloat16?
BFloat16 precision reduces memory usage by approximately 50% while maintaining better numerical stability than FP16, critical for fitting large models on consumer GPUs.

### Why Multiple Attempts?
Temperature variation generates diverse outputs. Lower temperatures (0.1) produce conservative results, while higher temperatures (0.3) explore alternative interpretations. The medical-grade selector chooses the highest quality attempt, not just the longest.

### Why LANCZOS Resampling?
LANCZOS provides the highest quality downsampling method, preserving text clarity better than other algorithms when resizing oversized images.

## Privacy and Security

- All processing occurs locally on your machine
- No external network calls after initial model download
- Files remain in original locations
- No data transmission to external servers
- Model weights stored in local Hugging Face cache

## Model License

This project uses the Qwen2.5-VL-7B-Instruct model. Please review the model license terms at:
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

## Acknowledgments

- Qwen Team at Alibaba Cloud for the Qwen2.5-VL models
- Hugging Face for model hosting and Transformers library
- PyMuPDF team for efficient PDF processing capabilities

## Support

For issues, questions, or feature requests, please open an issue on the project repository.

---

**Implementation**: Jupyter Notebook-based OCR system
**Model**: Qwen2.5-VL-7B-Instruct
**Framework**: PyTorch + Transformers
**Last Updated**: 2025-01-15
