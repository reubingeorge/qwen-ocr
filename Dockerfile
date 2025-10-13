# Qwen3-VL-30B-A3B-Thinking-FP8 OCR CLI Dockerfile
# This image provides GPU-accelerated OCR conversion using vLLM

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY ocr_converter.py .

# Create logs directory
RUN mkdir -p /app/logs

# Set executable permissions
RUN chmod +x ocr_converter.py

# Default command
ENTRYPOINT ["python3", "ocr_converter.py"]
CMD ["--help"]
