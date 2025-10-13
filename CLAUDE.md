## Key Differentiators

### ✅ Mandatory GPU Verification (Pre-Flight Check)
Before any model loading or processing, the application uses **nvidia-ml-py (pynvml)** to:
- Detect GPU and accurately measure available VRAM
- Verify minimum 20GB VRAM requirement
- Fail fast with clear error messages if requirements not met
- Prevent wasted time and OOM errors during model loading

### 🧠 Native Thinking Model with Chain-of-Thought (CoT) Reasoning
Uses the **Qwen3-VL-30B-A3B-Thinking-FP8 model** which has **built-in thinking capabilities**. Combined with CoT prompting, it guides the VLM through a step-by-step reasoning process:
1. Document analysis and layout detection
2. Text region identification and hierarchy
3. Careful extraction with formatting preservation
4. Self-verification against visual layout

The Thinking model's native reasoning architecture makes it exceptionally well-suited for complex OCR tasks, resulting in **significantly higher accuracy** compared to standard models, especially for documents with tables, forms, and multi-column layouts.

### ⚡ vLLM-Optimized Inference
- Uses **vLLM** for optimized inference (Transformers cannot load FP8 weights)
- Efficient GPU memory management with configurable utilization
- Multi-GPU support via tensor parallelism
- Faster inference compared to standard loading methods

### 🔄 Intelligent Multi-Retry Mechanism
Each page can be processed up to 3 times with progressively enhanced strategies:
- **Attempt 1**: Standard CoT inference leveraging thinking capabilities
- **Attempt 2**: Enhanced CoT with focused attention on low-confidence areas
- **Attempt 3**: Image enhancement + most detailed CoT prompt

The system automatically selects the best result across all attempts based on confidence scoring.

### 📂 In-Place File Generation
Text files are created in the exact same folder as their source PDFs, making organization effortless.# Qwen3-VL-30B-A3B-Thinking-FP8 CLI PDF-to-Text OCR Converter

## Project Summary

This is an intelligent **command-line PDF-to-Text conversion tool** that uses the **Qwen3-VL-30B-A3B-Thinking-FP8 model** - a state-of-the-art FP8-quantized Vision Language Model with native thinking capabilities - to extract text from PDF documents with exceptional accuracy. The system employs Chain-of-Thought (CoT) reasoning, vLLM-optimized inference, and an aggressive multi-retry mechanism to achieve maximum OCR quality.

**Core Functionality**: Command-line tool that recursively converts all PDFs in a folder to text files
- **Input**: A folder path provided via terminal command
- **Processing**: Recursively scans all subfolders, processes all PDFs using FP8 Thinking model
- **Output**: Plain text files (.txt) created in the SAME location as each PDF
- **Example**: PDF at `/docs/invoices/2024/march.pdf` → Text at `/docs/invoices/2024/march.txt`

**Model Information**:
- **Model**: Qwen3-VL-30B-A3B-Thinking-FP8 (FP8 Quantized - ONLY THIS MODEL)
  - HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
  - VRAM Required: **Minimum 20GB** (typically uses 20-28GB)
  - Recommended GPUs: RTX 4090 (24GB), RTX 6000 Ada (48GB), A5000 (24GB), A100 (40GB/80GB)
  - Deployment: **Requires vLLM** (Transformers doesn't support FP8 weights directly)
  - Quantization: Fine-grained FP8 with block size of 128
  - Performance: Nearly identical to BF16 original model

**CRITICAL**: Application performs **mandatory GPU verification** using nvidia-ml-py before attempting to load the model. If GPU doesn't meet minimum requirements, application exits gracefully with clear error message.

## Key Differentiators

### 🧠 Chain-of-Thought (CoT) Reasoning
Unlike traditional OCR systems, this application uses CoT prompting to guide the VLM through a step-by-step reasoning process:
1. Document analysis and layout detection
2. Text region identification and hierarchy
3. Careful extraction with formatting preservation
4. Self-verification against visual layout

This results in **significantly higher accuracy**, especially for complex documents with tables, forms, and multi-column layouts.

### 🔄 Intelligent Multi-Retry Mechanism
Each page can be processed up to 3 times with progressively enhanced strategies:
- **Attempt 1**: Standard CoT inference with comprehensive reasoning
- **Attempt 2**: Enhanced CoT with focused attention on low-confidence areas
- **Attempt 3**: Image enhancement + most detailed CoT prompt

The system automatically selects the best result across all attempts based on confidence scoring.

### ⚡ Auto-Model Selection
The application automatically detects available GPU VRAM and selects the optimal Qwen3-VL model:
- < 6GB VRAM → Qwen3-VL-2B-Instruct
- 6-12GB VRAM → Qwen3-VL-4B-Instruct
- 12-20GB VRAM → Qwen3-VL-8B-Instruct
- 20-40GB VRAM → Qwen3-VL-14B-Instruct
- 40GB+ VRAM → Qwen3-VL-72B-Instruct

### 📂 In-Place File Generation
Text files are created in the exact same folder as their source PDFs, making organization effortless.

## Architecture Overview

### Technology Stack
- **Language**: Python 3.10+
- **CLI Framework**: argparse or click for command-line interface - **Use latest stable version**
- **Inference Engine**: vLLM - **REQUIRED** (Transformers cannot load FP8 weights) - **Use latest stable version**
- **ML Framework**: PyTorch 2.0+ with CUDA - **Use latest stable version**
- **VLM Model**: Qwen3-VL-30B-A3B-Thinking-FP8 (HuggingFace) - **Use latest stable version**
- **GPU Detection**: nvidia-ml-py (pynvml) - **REQUIRED for VRAM verification** - **Use latest stable version**
- **Containerization**: Docker with GPU support
- **PDF Processing**: PyMuPDF (fitz) or pdf2image - **Use latest stable version**
- **Image Enhancement**: OpenCV - **Use latest stable version**
- **Vision Utils**: qwen-vl-utils (0.0.14+) - **REQUIRED** - **Use latest stable version**
- **Progress Display**: tqdm for progress bars - **Use latest stable version**

> **Note**: All package versions listed are minimums. Always install the latest stable releases using `pip install --upgrade <package>` or `pip search <package>` to find the newest version.

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface                            │
│  Command: python ocr_converter.py /path/to/folder          │
│  Arguments: --skip-existing --verbose --model, etc.        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Folder Scanner                            │
│  • Recursive directory walking                              │
│  • PDF file discovery                                       │
│  • Skip-existing logic                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PDF-to-Text Converter                       │
│  • PDF → Image Conversion (per page)                       │
│  • Page-by-Page Processing                                  │
│  • Text Aggregation                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     OCR Engine (Core)                        │
│  • CoT Prompt Generation                                    │
│  • Qwen3-VL Inference                                       │
│  • Retry Logic (up to 3 attempts)                          │
│  • Confidence Scoring                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Text File Generator                        │
│  • Create .txt in same folder as PDF                       │
│  • Preserve formatting                                      │
│  • Add optional metadata                                    │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
qwen3-ocr-cli/
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Container definition with GPU support
├── requirements.txt            # Python dependencies (latest versions)
├── README.md                   # User documentation
├── ocr_converter.py           # Main CLI entry point
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── cli.py                 # CLI interface and orchestration
│   ├── gpu_detector.py        # GPU/VRAM detection
│   ├── model_selector.py      # Auto model selection
│   ├── model_manager.py       # Model loading and caching
│   ├── pdf_scanner.py         # Folder walking and PDF discovery
│   ├── ocr_engine.py          # Core OCR with CoT & retry
│   ├── text_generator.py      # Text file creation
│   ├── cot_prompts.py         # CoT prompt templates
│   └── confidence_scorer.py   # Confidence calculation
│
└── logs/                       # Processing logs
    └── ocr.log
```

## Installation and Setup

> **⚠️ IMPORTANT**: Always use the **latest stable versions** of all packages. Use `pip search <package_name>` to find the most recent versions before installation. This ensures compatibility with the latest Qwen3-VL models and best performance.

### Prerequisites
```bash
# System requirements
- Linux OS (Ubuntu 20.04+ recommended) or macOS
- NVIDIA GPU with CUDA 11.8+ (for GPU acceleration)
- Docker and Docker Compose (optional)
- nvidia-docker2 (if using Docker)

# Install nvidia-docker2 (Linux only)
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Quick Start with Docker

1. **Clone the repository**
```bash
git clone <repository-url>
cd qwen3-ocr-cli
```

2. **Build Docker container**
```bash
docker build -t qwen3-ocr-cli .
```

3. **Run the application**
```bash
# Basic usage
docker run --gpus all -v /path/to/your/documents:/data qwen3-ocr-cli /data

# With options
docker run --gpus all -v /path/to/your/documents:/data qwen3-ocr-cli /data --skip-existing --verbose
```

### Manual Installation (Non-Docker)

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# IMPORTANT: Use latest package versions
# Search for latest versions before installing
pip search vllm              # Check latest vLLM version
pip search nvidia-ml-py      # Check latest nvidia-ml-py version
pip search transformers      # Check latest transformers version
pip search torch             # Check latest PyTorch version

# Install dependencies with latest versions
pip install --upgrade pip

# Install all packages with latest versions (CRITICAL PACKAGES)
pip install --upgrade vllm qwen-vl-utils nvidia-ml-py transformers torch torchvision pillow opencv-python pymupdf tqdm click

# Or use requirements.txt (ensure it has latest versions)
pip install -r requirements.txt

# Verify installation
python ocr_converter.py --help

# Test GPU verification
python -c "import pynvml; pynvml.nvmlInit(); print('GPU detection working!')"
```

**Note**: Always use the latest stable versions of packages to get:
- Latest Qwen3-VL model support
- vLLM performance improvements
- Security patches
- Bug fixes
- New features

**Critical Dependencies**:
- `vllm` - REQUIRED for model loading (Transformers cannot load FP8)
- `qwen-vl-utils` (0.0.14+) - REQUIRED for vision processing
- `nvidia-ml-py` - REQUIRED for GPU verification
- `transformers` - For processor only (not for model loading)

## Usage Guide

### Basic Command Structure

```bash
python ocr_converter.py <folder_path> [options]
```

### Examples

**1. Process all PDFs in a folder**
```bash
python ocr_converter.py /home/user/documents
```

**2. Skip PDFs that already have text files**
```bash
python ocr_converter.py /home/user/documents --skip-existing
```

**3. Use specific model (override auto-detection)**
```bash
python ocr_converter.py /home/user/documents --model Qwen3-VL-8B-Instruct
```

**4. Verbose output with detailed logging**
```bash
python ocr_converter.py /home/user/documents --verbose
```

**5. Dry run (see what would be processed)**
```bash
python ocr_converter.py /home/user/documents --dry-run
```

**6. Filter specific PDFs by pattern**
```bash
python ocr_converter.py /home/user/documents --file-pattern "invoice*.pdf"
```

**7. Adjust confidence threshold**
```bash
python ocr_converter.py /home/user/documents --confidence 0.90
```

**8. Reduce maximum retry attempts**
```bash
python ocr_converter.py /home/user/documents --max-retries 2
```

**9. Exclude metadata from output files**
```bash
python ocr_converter.py /home/user/documents --no-metadata
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `folder` | Path to folder containing PDFs (required) | - |
| `--confidence` | Confidence threshold for retry (0.0-1.0) | `0.85` |
| `--max-retries` | Maximum retry attempts per page | `3` |
| `--skip-existing` | Skip PDFs with existing .txt files | `False` |
| `--dry-run` | Show what would be processed | `False` |
| `--file-pattern` | Filter PDFs by name pattern | `None` |
| `-v, --verbose` | Detailed logging output | `False` |
| `--no-metadata` | Exclude metadata from text files | `False` |
| `--gpu-memory` | GPU memory utilization (0.1-0.95) | `0.70` |

### Example Output

```bash
$ python ocr_converter.py /home/user/documents

Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter
========================================================

Step 1: Verifying GPU Compatibility...
✓ GPU VERIFICATION PASSED
==================================================
GPU: NVIDIA RTX 4090
Total VRAM: 24.0GB
Free VRAM: 23.2GB
Used VRAM: 0.8GB
✓ Sufficient VRAM for Qwen3-VL-30B-A3B-Thinking-FP8

Step 2: Scanning folder structure...
Found 47 PDF files in 12 subfolders
Skipping 12 PDFs (text files already exist)

Step 3: Loading model (using vLLM)...
✓ Model loaded successfully (23.8GB VRAM used)

Step 4: Processing 35 PDFs...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 35/35

Current: /home/user/documents/reports/2024/Q1/report.pdf
Pages: ████████████░░░░░░░░ 12/20 | Attempt: 1/3 | Confidence: 0.89

Summary:
--------
Model: Qwen3-VL-30B-A3B-Thinking-FP8
GPU: NVIDIA RTX 4090 (24GB)
Total PDFs processed: 35
Total pages: 678
Average confidence: 0.91
Pages requiring retry: 92 (13.6%)
Failed conversions: 0
Total time: 1h 45m 22s

✓ All PDFs converted successfully!
```

## Input/Output Structure Example

### Before Processing
```
/home/user/documents/
├── report.pdf
├── invoices/
│   ├── january.pdf
│   ├── february.pdf
│   └── march.pdf
└── contracts/
    └── 2024/
        ├── contract_a.pdf
        └── contract_b.pdf
```

### After Processing
```
/home/user/documents/
├── report.pdf
├── report.txt                    ← Created
├── invoices/
│   ├── january.pdf
│   ├── january.txt               ← Created
│   ├── february.pdf
│   ├── february.txt              ← Created
│   ├── march.pdf
│   └── march.txt                 ← Created
└── contracts/
    └── 2024/
        ├── contract_a.pdf
        ├── contract_a.txt        ← Created
        ├── contract_b.pdf
        └── contract_b.txt        ← Created
```

## Text File Output Format

Each PDF generates one text file with this structure:

```
========================================
Document: invoice_march_2024.pdf
Source Path: /home/user/documents/invoices/invoice_march_2024.pdf
Processed: 2025-01-15 14:30:00
Pages: 5
Average Confidence: 92.5%
Model: Qwen3-VL-4B-Instruct
========================================

========== PAGE 1 ==========

Invoice #12345
Date: March 15, 2024

Bill To:
John Smith
123 Main Street
...

========== PAGE 2 ==========

Item Description    Quantity    Price
Widget A                5       $10.00
Widget B                3       $15.00
...

========== PAGE 5 ==========

Total Amount Due: $125.00

---
Generated by Qwen3-VL OCR CLI
Processing Time: 45.2 seconds
Retries: 2 pages required retry
```

## GPU Requirements and Performance

### Minimum Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: **20GB minimum** (MANDATORY - verified before model loading)
- **CUDA**: Version 11.8 or higher
- **Docker**: GPU runtime support (if using Docker)

### Recommended GPUs
- ✅ **RTX 4090** (24GB) - Excellent choice
- ✅ **RTX 6000 Ada** (48GB) - Excellent choice
- ✅ **A5000** (24GB) - Good choice
- ✅ **A100** (40GB/80GB) - Excellent choice
- ✅ **H100** (80GB) - Excellent choice
- ❌ **RTX 3090** (24GB) - May work but tight on memory
- ❌ **RTX 3080** (10GB/12GB) - Insufficient VRAM
- ❌ **GTX 1080 Ti** (11GB) - Insufficient VRAM

**GPU Verification**: Application uses **nvidia-ml-py (pynvml)** to verify GPU compatibility BEFORE attempting to load the model. If your GPU has less than 20GB VRAM, the application will exit with a clear error message.

### Expected Performance

**Processing Speed** (per page with FP8 + vLLM):
- **Best case** (1 attempt, high confidence): 5-10 seconds
- **Average case** (1-2 attempts): 10-20 seconds  
- **Worst case** (3 attempts with enhancement): 25-40 seconds

**Batch Processing Examples**:
- 10-page PDF: 1-5 minutes (avg 2-3 minutes)
- 50-page PDF: 8-25 minutes (avg 12-15 minutes)
- 100 PDFs (10 pages each): 3-8 hours (avg 5 hours)
- 1000 PDFs: 30-80 hours (avg 50 hours)

*Performance varies based on document complexity and GPU model.*

### VRAM Usage
- **Model Loading**: ~22-24GB
- **During Inference**: ~22-28GB (with 70% GPU memory utilization)
- **Peak Usage**: Up to 28GB with overhead
- **Configurable**: Use `--gpu-memory 0.60` to reduce utilization if needed

## Configuration Options

### Environment Variables

```bash
# Set in terminal or .env file

# GPU Settings
export CUDA_VISIBLE_DEVICES=0              # Which GPU to use
export GPU_MEMORY_FRACTION=0.9             # Max GPU memory to use

# Model Settings  
export DEFAULT_MODEL=auto                  # or specific model name
export CONFIDENCE_THRESHOLD=0.85           # Retry threshold
export MAX_RETRIES=3                       # Maximum retry attempts

# Performance
export MAX_TOKENS=4096                     # Max tokens for CoT generation
export TIMEOUT_SECONDS=60                  # Per-attempt timeout

# Output
export INCLUDE_METADATA=true               # Add metadata to text files
export ADD_PAGE_MARKERS=true               # Show page numbers in output
```

### Using with Docker

```bash
# Pass environment variables
docker run --gpus all \
  -e CONFIDENCE_THRESHOLD=0.90 \
  -e MAX_RETRIES=2 \
  -v /path/to/docs:/data \
  qwen3-ocr-cli /data

# Or use docker-compose
# Edit docker-compose.yml with your settings
docker-compose run ocr-cli /data
```

## Troubleshooting

### Common Issues

**GPU Not Detected / Verification Fails**
```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Verify nvidia-ml-py is working
python -c "import pynvml; pynvml.nvmlInit(); print('OK')"

# If nvidia-ml-py not installed
pip install --upgrade nvidia-ml-py

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Insufficient VRAM Error**
```
# Application will show:
❌ GPU VERIFICATION FAILED
GPU: NVIDIA RTX 3080
VRAM: 10.0GB (Need: 20GB)

# Solutions:
1. Use a GPU with at least 20GB VRAM
2. Upgrade to RTX 4090 (24GB), A5000 (24GB), or better
3. Use a cloud GPU instance (AWS, GCP, Lambda Labs)
```

**vLLM Installation Issues**
```bash
# Install/reinstall vLLM
pip install --upgrade vllm

# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# If CUDA version mismatch:
pip install vllm --force-reinstall --no-cache-dir

# For specific CUDA version (example for CUDA 11.8):
pip install vllm-cuda118
```

**Out of Memory (OOM) During Inference**
```bash
# Reduce GPU memory utilization
python ocr_converter.py /data --gpu-memory 0.60

# Clear GPU cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Check what's using GPU memory
nvidia-smi

# Kill other GPU processes if needed
sudo kill -9 <PID>
```

**Model Download Issues**
```bash
# Manually download model first
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-30B-A3B-Thinking-FP8')"

# Check HuggingFace cache
ls ~/.cache/huggingface/hub/

# Clear cache and redownload
rm -rf ~/.cache/huggingface/hub/models--Qwen*
```

**Slow Processing / Performance Issues**
```bash
# Verify GPU is being used
nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

# Check if vLLM is loaded correctly
python -c "from vllm import LLM; print('vLLM OK')"

# Monitor VRAM usage
watch -n 1 nvidia-smi

# Reduce retries for testing
python ocr_converter.py /data --max-retries 1
```

**qwen-vl-utils Not Found**
```bash
# Install required version
pip install "qwen-vl-utils>=0.0.14"

# Check version
python -c "import qwen_vl_utils; print(qwen_vl_utils.__version__)"
```

**Poor OCR Accuracy**
```bash
# Increase confidence threshold (more retries)
python ocr_converter.py /data --confidence 0.90

# Use larger model
python ocr_converter.py /data --model Qwen3-VL-8B-Instruct

# Check verbose output for issues
python ocr_converter.py /data --verbose
```

**Permission Errors**
```bash
# Check folder permissions
ls -la /path/to/documents

# Run with appropriate user permissions
sudo python ocr_converter.py /data  # Not recommended

# Better: Fix folder permissions
sudo chown -R $USER:$USER /path/to/documents
```

**Files Not Being Processed**
```bash
# Use dry-run to see what would be processed
python ocr_converter.py /data --dry-run

# Check if being skipped (already have .txt files)
python ocr_converter.py /data --verbose

# Remove --skip-existing flag
python ocr_converter.py /data  # Will overwrite existing .txt files
```

## Logging and Monitoring

### Log Files
```bash
# Application logs
logs/ocr.log                  # Main processing log

# View logs in real-time
tail -f logs/ocr.log

# View last 100 lines
tail -n 100 logs/ocr.log

# Search for errors
grep "ERROR" logs/ocr.log
```

### Verbose Mode
```bash
# Enable detailed console output
python ocr_converter.py /data --verbose

# Shows:
# - Each PDF being processed
# - Page-by-page progress
# - Retry attempts and reasons
# - Confidence scores
# - Model inference details
# - File creation confirmations
```

## Best Practices

### For Large Batches

1. **Start with dry-run**
   ```bash
   python ocr_converter.py /data --dry-run
   ```
   Verify what will be processed before starting.

2. **Use skip-existing for resumability**
   ```bash
   python ocr_converter.py /data --skip-existing
   ```
   If interrupted, rerun and it will continue from where it stopped.

3. **Test with small subset first**
   ```bash
   python ocr_converter.py /data/test_folder --verbose
   ```
   Validate accuracy and performance before processing everything.

4. **Run overnight for large batches**
   ```bash
   nohup python ocr_converter.py /data > output.log 2>&1 &
   ```
   Runs in background, survives terminal disconnection.

5. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```
   Keep an eye on GPU utilization and memory.

### For Best Accuracy

1. **Use appropriate model size**
   - Complex documents → 8B or 14B model
   - Simple documents → 4B model is sufficient

2. **Adjust confidence threshold**
   ```bash
   python ocr_converter.py /data --confidence 0.90
   ```
   Higher threshold = more retries = better accuracy (but slower)

3. **Check verbose output for low-confidence pages**
   ```bash
   python ocr_converter.py /data --verbose | grep "confidence"
   ```

4. **Manually review low-confidence conversions**
   Look for pages that required 3 retry attempts.

## Maintenance

### Regular Tasks

```bash
# Clean up log files (run weekly)
rm logs/ocr.log
# Or truncate
: > logs/ocr.log

# Check disk space
df -h

# Update models (download latest versions)
pip install --upgrade transformers

# Clear model cache to force re-download
rm -rf ~/.cache/huggingface/hub/models--Qwen*
```

### Updating the Application

```bash
# Update dependencies
pip install --upgrade transformers torch pillow opencv-python

# Verify everything still works
python ocr_converter.py --help

# Test with sample PDF
python ocr_converter.py /path/to/test/folder --verbose
```

## Localhost Deployment Notes

This application is designed for **local CLI deployment only**. 

### Simple Setup
- Runs from terminal: `python ocr_converter.py /path/to/folder`
- No web server or authentication needed
- Direct file system access
- All processing happens on your local GPU
- Files stored locally in their original locations

### Data Handling
- **File storage**: Text files created alongside original PDFs in same folders
- **Privacy**: No external network calls - everything stays on your machine
- **Cleanup**: Manually delete text files if needed (same folders as PDFs)
- **Organization**: Files stay organized in your existing folder structure

## Performance Optimization Tips

1. **Use skip-existing for interrupted jobs**
   ```bash
   python ocr_converter.py /data --skip-existing
   ```
   Dramatically speeds up reprocessing.

2. **Adjust confidence threshold based on needs**
   - Need speed? `--confidence 0.80`
   - Need accuracy? `--confidence 0.90`

3. **Use appropriate model for task**
   - Simple scans: 2B or 4B model
   - Complex documents: 8B model
   - Best quality: 14B model (if VRAM allows)

4. **Process in batches**
   - Don't process thousands of PDFs at once
   - Break into manageable chunks (100-500 PDFs)

5. **Monitor and restart if needed**
   - Check progress periodically
   - GPU might need clearing: restart the process

## Support and Contact

For issues, questions, or feature requests:
- GitHub Issues: [Repository URL]
- Documentation: [Docs URL]
- Email: [Contact email]

## Acknowledgments

- **Qwen Team**: For the exceptional Qwen3-VL models
- **HuggingFace**: For the Transformers library
- **Open Source Community**: For all the supporting libraries

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Type**: Command-Line Interface (CLI)  
**Deployment**: Localhost only