# Qwen3-VL CLI PDF-to-Text OCR Converter

An intelligent command-line PDF-to-Text conversion tool powered by Qwen3-VL Vision Language Models with Chain-of-Thought reasoning and multi-retry mechanism for exceptional OCR accuracy.

## Features

- **Chain-of-Thought (CoT) Reasoning**: Advanced multi-step reasoning process for superior text extraction accuracy
- **Intelligent Multi-Retry**: Up to 3 attempts per page with progressively enhanced strategies
- **Auto-Model Selection**: Automatically selects optimal model based on available GPU VRAM
- **Recursive Processing**: Processes all PDFs in a folder and its subfolders
- **In-Place Output**: Creates .txt files in the same location as source PDFs
- **Confidence Scoring**: Automatic quality assessment and retry logic
- **Docker Support**: GPU-enabled containerization for easy deployment

## Quick Start

### Using Docker (Recommended)

```bash
# 1. Build the image
docker build -t qwen3-ocr-cli .

# 2. Run on a folder
docker run --gpus all -v /path/to/your/documents:/data qwen3-ocr-cli /data

# 3. With options
docker run --gpus all -v /path/to/your/documents:/data qwen3-ocr-cli /data --skip-existing --verbose
```

### Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the converter
python ocr_converter.py /path/to/documents
```

## Usage

### Basic Commands

```bash
# Process all PDFs in a folder
python ocr_converter.py /path/to/folder

# Skip PDFs that already have .txt files
python ocr_converter.py /path/to/folder --skip-existing

# Use specific model
python ocr_converter.py /path/to/folder --model Qwen/Qwen3-VL-30B-A3B-Instruct

# Verbose output
python ocr_converter.py /path/to/folder --verbose

# Dry run (see what would be processed)
python ocr_converter.py /path/to/folder --dry-run
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `folder` | Path to folder containing PDFs (required) | - |
| `--model` | Model name or 'auto' for auto-detection | `auto` |
| `--confidence` | Confidence threshold for retry (0.0-1.0) | `0.85` |
| `--max-retries` | Maximum retry attempts per page | `3` |
| `--skip-existing` | Skip PDFs with existing .txt files | `False` |
| `--dry-run` | Show what would be processed | `False` |
| `--file-pattern` | Filter PDFs by name pattern | `*.pdf` |
| `-v, --verbose` | Detailed logging output | `False` |
| `--no-metadata` | Exclude metadata from text files | `False` |

### Examples

**1. Basic processing:**
```bash
python ocr_converter.py ~/Documents
```

**2. Resume interrupted job:**
```bash
python ocr_converter.py ~/Documents --skip-existing
```

**3. High accuracy mode:**
```bash
python ocr_converter.py ~/Documents --confidence 0.90
```

**4. Process specific PDFs:**
```bash
python ocr_converter.py ~/Documents --file-pattern "invoice*.pdf"
```

## Architecture

### Project Structure

```
qwen-ocr/
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── CLAUDE.md                   # Project specification
├── ocr_converter.py           # Main entry point
├── src/                        # Source code
│   ├── __init__.py
│   ├── cli.py                 # CLI interface
│   ├── gpu_detector.py        # GPU/VRAM detection
│   ├── model_selector.py      # Model selection logic
│   ├── model_manager.py       # Model loading/caching
│   ├── pdf_scanner.py         # PDF discovery
│   ├── ocr_engine.py          # Core OCR engine
│   ├── text_generator.py      # Text file generation
│   ├── cot_prompts.py         # CoT prompt templates
│   └── confidence_scorer.py   # Quality assessment
└── logs/                       # Processing logs
```

### Technology Stack

- **Python 3.10+**
- **PyTorch 2.8.0+** with CUDA support
- **Transformers 4.57.0+** for Qwen3-VL models
- **PyMuPDF 1.26.5+** for PDF processing
- **OpenCV 4.12.0+** for image enhancement
- **Click 8.3.0+** for CLI interface

## Requirements

### Minimum System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows WSL2
- **GPU**: NVIDIA GPU with 20GB+ VRAM (for Qwen3-VL models)
- **CUDA**: Version 11.8 or higher
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space (for models)

### GPU Requirements by Model

| Model | VRAM Required | Performance |
|-------|---------------|-------------|
| Qwen3-VL-30B-A3B-Instruct-FP8 | 12-24 GB | Quantized 30B MoE - Good balance (FP8) |
| Qwen3-VL-30B-A3B-Instruct | 24-48 GB | Full precision 30B MoE - High quality |
| Qwen3-VL-235B-A22B-Instruct-FP8 | 48-80 GB | Quantized 235B MoE - Excellent (FP8) |
| Qwen3-VL-235B-A22B-Instruct | 80GB+ | Full precision 235B MoE - Best quality (A100/H100 required) |

## Installation

### Prerequisites

**Ubuntu/Debian:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**NVIDIA Docker (for Docker usage):**
```bash
# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Setup Virtual Environment

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Docker Usage

### Building the Image

```bash
docker build -t qwen3-ocr-cli .
```

### Running with Docker

**Basic usage:**
```bash
docker run --gpus all -v /path/to/docs:/data qwen3-ocr-cli /data
```

**With environment variables:**
```bash
docker run --gpus all \
  -e CONFIDENCE_THRESHOLD=0.90 \
  -e MAX_RETRIES=2 \
  -v /path/to/docs:/data \
  qwen3-ocr-cli /data --verbose
```

**Using docker-compose:**
```bash
# Edit docker-compose.yml to set your document path
docker-compose run qwen-ocr /data
```

## Output Format

Each PDF generates a .txt file in the same location:

```
========================================
Document: invoice.pdf
Source Path: /home/user/docs/invoice.pdf
Processed: 2025-10-13 14:30:00
Pages: 3
Average Confidence: 92.5%
Model: Qwen/Qwen3-VL-30B-A3B-Instruct
========================================

========== PAGE 1 ==========

Invoice #12345
Date: October 13, 2025

Bill To:
John Smith
...

========== PAGE 2 ==========
...

---
Generated by Qwen3-VL OCR CLI
Processing Time: 45.2 seconds
Retries: 1 pages required retry
```

## Performance

**Processing Speed** (per page):
- Best case: 2-5 seconds
- Average: 5-12 seconds
- Worst case: 15-25 seconds

**Batch Examples:**
- 10-page PDF: 30s - 4min
- 100 PDFs (10 pages each): 1-3 hours

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Verify CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

```bash
# Use quantized FP8 model (uses less VRAM)
python ocr_converter.py /data --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Poor OCR Quality

```bash
# Increase confidence threshold (more retries)
python ocr_converter.py /data --confidence 0.90

# Use larger model (if you have 48GB+ VRAM)
python ocr_converter.py /data --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```

## Development

### Running Tests

```bash
# Test individual modules
python -m src.gpu_detector
python -m src.model_selector
python -m src.confidence_scorer
```

### Logging

Logs are written to:
- Console (INFO level by default)
- `logs/ocr.log` (all levels)

Enable verbose logging:
```bash
python ocr_converter.py /data --verbose
```

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

For issues and questions:
- GitHub Issues: [Repository URL]
- Documentation: See CLAUDE.md for detailed specifications

## Acknowledgments

- **Qwen Team**: For the exceptional Qwen3-VL models
- **HuggingFace**: For the Transformers library
- **Open Source Community**: For supporting libraries

---

**Version**: 1.0.0
**Last Updated**: 2025-10-13
**Author**: Reubin George
