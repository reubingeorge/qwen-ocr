"""
GPU Detection Module

Detects NVIDIA GPU availability and VRAM capacity.
"""

import logging
from typing import Tuple, Optional

try:
    import torch
    import pynvml
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detects and provides information about available GPUs."""

    def __init__(self):
        self.gpu_available = False
        self.gpu_name = None
        self.vram_gb = 0
        self.compute_capability = 0.0
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect GPU and VRAM."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. GPU detection disabled.")
            return

        if not torch.cuda.is_available():
            logger.warning("No CUDA-capable GPU detected.")
            return

        try:
            # Initialize NVML
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                logger.warning("No NVIDIA GPUs found.")
                return

            # Get first GPU info (device 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(handle)

            # Get VRAM in bytes and convert to GB
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.vram_gb = mem_info.total / (1024 ** 3)

            # Get compute capability using PyTorch
            capability = torch.cuda.get_device_capability(0)
            self.compute_capability = float(f"{capability[0]}.{capability[1]}")

            self.gpu_available = True
            logger.info(f"GPU detected: {self.gpu_name} with {self.vram_gb:.1f}GB VRAM (Compute {self.compute_capability})")

            # Cleanup
            pynvml.nvmlShutdown()

        except Exception as e:
            logger.error(f"Error detecting GPU: {e}")
            self.gpu_available = False

    def get_gpu_info(self) -> Tuple[bool, Optional[str], float, float]:
        """
        Get GPU information.

        Returns:
            Tuple of (gpu_available, gpu_name, vram_gb, compute_capability)
        """
        return self.gpu_available, self.gpu_name, self.vram_gb, self.compute_capability

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available

    def get_vram_gb(self) -> float:
        """Get VRAM in GB."""
        return self.vram_gb

    def get_compute_capability(self) -> float:
        """Get GPU compute capability."""
        return self.compute_capability

    def supports_fp8(self) -> bool:
        """Check if GPU supports FP8 (requires compute capability >= 8.9)."""
        return self.compute_capability >= 8.9

    def get_device(self) -> str:
        """
        Get PyTorch device string.

        Returns:
            'cuda' if GPU available, 'cpu' otherwise
        """
        if self.gpu_available and TORCH_AVAILABLE and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'


def detect_gpu() -> GPUDetector:
    """
    Convenience function to detect GPU.

    Returns:
        GPUDetector instance
    """
    return GPUDetector()


if __name__ == "__main__":
    # Test GPU detection
    logging.basicConfig(level=logging.INFO)
    detector = detect_gpu()
    available, name, vram, compute_cap = detector.get_gpu_info()

    if available:
        print(f"✓ GPU detected: {name}")
        print(f"✓ VRAM: {vram:.1f}GB")
        print(f"✓ Compute Capability: {compute_cap}")
        print(f"✓ Supports FP8: {detector.supports_fp8()}")
        print(f"✓ Device: {detector.get_device()}")
    else:
        print("✗ No GPU available")
        print(f"✗ Device: {detector.get_device()}")
