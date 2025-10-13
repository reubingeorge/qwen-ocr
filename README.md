# Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter

An intelligent **command-line PDF-to-Text conversion tool** powered by the **Qwen3-VL-30B-A3B-Thinking-FP8 model** - a state-of-the-art Vision Language Model with native thinking capabilities and FP8 quantization for optimal performance.

## 🚀 Key Features

### 🧠 Native Thinking Model with Chain-of-Thought Reasoning
Uses the **Qwen3-VL-30B-A3B-Thinking-FP8 model** with built-in thinking capabilities and CoT prompting to guide step-by-step reasoning for exceptional OCR accuracy.

### ✅ Mandatory GPU Verification
Pre-flight checks using **nvidia-ml-py** to verify 20GB+ VRAM before model loading, preventing wasted time and OOM errors.

### ⚡ vLLM-Optimized Inference
Leverages **vLLM** for optimized FP8 inference with efficient GPU memory management and multi-GPU support.

### 🔄 Intelligent Multi-Retry Mechanism
Automatically retries low-confidence pages (up to 3 attempts) with progressively enhanced strategies and image preprocessing.

### 📂 In-Place File Generation
Text files created in the same folder as source PDFs for effortless organization.

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: **Minimum 20GB** (verified before model loading)
- **Recommended GPUs**: RTX 4090 (24GB), RTX 6000 Ada (48GB), A5000 (24GB), A100 (40GB/80GB)

### Software
- Python 3.10+
- CUDA 11.8+
- NVIDIA drivers with GPU support
- Docker (optional, for containerized deployment)

## 🔧 Installation

### Option 1: Quick Start with pip

```bash
# Clone repository
git clone <repository-url>
cd qwen-ocr

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python ocr_converter.py --help
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t qwen3-ocr-cli .

# Run with Docker
docker run --gpus all -v /path/to/documents:/data qwen3-ocr-cli /data

# Or use docker-compose
docker-compose build
docker-compose run qwen-ocr /data --skip-existing
```

## 📖 Usage

### Basic Usage

```bash
# Process all PDFs in a directory
python ocr_converter.py /path/to/documents

# Skip PDFs that already have text files
python ocr_converter.py /path/to/documents --skip-existing

# Verbose output
python ocr_converter.py /path/to/documents --verbose

# Dry run (see what would be processed)
python ocr_converter.py /path/to/documents --dry-run
```

### Advanced Options

```bash
# Adjust confidence threshold (0.0-1.0)
python ocr_converter.py /path/to/documents --confidence 0.90

# Limit retry attempts
python ocr_converter.py /path/to/documents --max-retries 2

# Filter specific PDFs
python ocr_converter.py /path/to/documents --file-pattern "invoice*.pdf"

# Exclude metadata from output
python ocr_converter.py /path/to/documents --no-metadata

# Adjust GPU memory usage
python ocr_converter.py /path/to/documents --gpu-memory 0.60
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `folder` | Path to folder with PDFs (required) | - |
| `--confidence` | Confidence threshold (0.0-1.0) | `0.85` |
| `--max-retries` | Max retry attempts per page | `3` |
| `--skip-existing` | Skip PDFs with existing .txt files | `False` |
| `--dry-run` | Preview without processing | `False` |
| `--file-pattern` | Filter PDFs by pattern | `*.pdf` |
| `-v, --verbose` | Detailed logging | `False` |
| `--no-metadata` | Exclude metadata from output | `False` |
| `--gpu-memory` | GPU memory utilization (0.1-0.95) | `0.70` |

## 📁 Project Structure

```
qwen-ocr/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # CLI interface
│   ├── gpu_detector.py          # GPU/VRAM verification
│   ├── model_manager.py         # vLLM model management
│   ├── pdf_scanner.py           # Recursive PDF discovery
│   ├── ocr_engine.py            # Core OCR with CoT & retry
│   ├── text_generator.py        # Text file creation
│   ├── cot_prompts.py           # Chain-of-Thought prompts
│   └── confidence_scorer.py     # Confidence calculation
├── ocr_converter.py             # Main entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
└── README.md                    # This file
```

## 💡 Example Output

```bash
$ python ocr_converter.py ./documents

==============================================================
Qwen3-VL-30B-A3B-Thinking-FP8 PDF-to-Text OCR Converter
==============================================================

==============================================================
STEP 1: Verifying GPU Compatibility
==============================================================
GPU Name: NVIDIA RTX 4090
Total VRAM: 24.0 GB
Free VRAM: 23.2 GB
Required VRAM: 20.0 GB
✓ GPU VERIFICATION PASSED

==============================================================
STEP 2: Scanning for PDF Files
==============================================================
Total PDFs found: 15
PDFs to process: 15

==============================================================
STEP 3: Loading Model
==============================================================
Loading model: Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
✓ Model loaded successfully

==============================================================
STEP 4: Processing 15 PDF Files
==============================================================
Processing PDFs: 100%|████████████| 15/15

==============================================================
PROCESSING SUMMARY
==============================================================
Successfully processed: 15
Failed: 0
Total pages: 89
Total time: 45.2m
Average time per page: 30.5s

✓ All PDFs converted successfully!
```

## 🧪 Testing Components

Each module can be tested individually:

```bash
# Test GPU detection
python -m src.gpu_detector

# Test PDF scanner
python -m src.pdf_scanner /path/to/documents

# Test confidence scorer
python -m src.confidence_scorer

# Test CoT prompts
python -m src.cot_prompts
```

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify nvidia-ml-py
python -c "import pynvml; pynvml.nvmlInit(); print('OK')"
```

### Insufficient VRAM
```bash
# Reduce GPU memory utilization
python ocr_converter.py /data --gpu-memory 0.60

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### vLLM Installation Issues
```bash
# Reinstall vLLM
pip install --upgrade vllm --force-reinstall --no-cache-dir
```

### Slow Performance
```bash
# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

# Reduce retry attempts for testing
python ocr_converter.py /data --max-retries 1
```

## 📊 Performance

**Processing Speed** (per page):
- Best case: 5-10 seconds
- Average: 10-20 seconds
- Worst case (3 retries): 25-40 seconds

**VRAM Usage**:
- Model loading: ~22-24GB
- During inference: ~22-28GB
- Peak usage: Up to 28GB

## 🔐 Security & Privacy

- All processing happens **locally** on your machine
- No external network calls (except initial model download)
- Files stay in original locations
- No data leaves your system

## 📄 License

This project uses the Qwen3-VL-30B-A3B-Thinking-FP8 model which has its own license terms.
Please review at: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8

## 🙏 Acknowledgments

- **Qwen Team**: For the exceptional Qwen3-VL models
- **vLLM Team**: For optimized inference engine
- **HuggingFace**: For model hosting and Transformers library

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Version**: 1.0.0
**Model**: Qwen3-VL-30B-A3B-Thinking-FP8
**Last Updated**: 2025-01-15
