"""
GPU Detection and VRAM Verification Module

This module uses nvidia-ml-py (pynvml) to detect GPU capabilities and verify
that the system meets the minimum VRAM requirements for the Qwen3-VL-30B-A3B-Thinking-FP8 model.
"""

import sys
from typing import Dict, Optional
try:
    import pynvml
except ImportError:
    print("ERROR: nvidia-ml-py not installed. Install with: pip install nvidia-ml-py")
    sys.exit(1)


class GPUDetector:
    """Detects GPU capabilities and verifies VRAM requirements."""

    # Minimum VRAM requirement for Qwen3-VL-30B-A3B-Thinking-FP8 model
    MINIMUM_VRAM_GB = 20.0

    def __init__(self):
        """Initialize the GPU detector."""
        self.initialized = False
        try:
            pynvml.nvmlInit()
            self.initialized = True
        except Exception as e:
            print(f"ERROR: Failed to initialize NVML: {e}")
            print("Make sure NVIDIA drivers are installed and GPU is available.")
            sys.exit(1)

    def get_gpu_info(self, device_index: int = 0) -> Dict[str, any]:
        """
        Get detailed GPU information.

        Args:
            device_index: GPU device index (default: 0)

        Returns:
            Dictionary containing GPU information
        """
        if not self.initialized:
            raise RuntimeError("GPU detector not initialized")

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Convert to GB
            total_vram_gb = mem_info.total / (1024 ** 3)
            free_vram_gb = mem_info.free / (1024 ** 3)
            used_vram_gb = mem_info.used / (1024 ** 3)

            return {
                'name': name,
                'index': device_index,
                'total_vram_gb': total_vram_gb,
                'free_vram_gb': free_vram_gb,
                'used_vram_gb': used_vram_gb,
                'total_vram_bytes': mem_info.total,
                'free_vram_bytes': mem_info.free,
                'used_vram_bytes': mem_info.used
            }
        except Exception as e:
            print(f"ERROR: Failed to get GPU info: {e}")
            sys.exit(1)

    def verify_requirements(self, device_index: int = 0) -> bool:
        """
        Verify that the GPU meets minimum VRAM requirements.

        Args:
            device_index: GPU device index (default: 0)

        Returns:
            True if requirements are met, False otherwise
        """
        gpu_info = self.get_gpu_info(device_index)

        print(f"\n{'='*50}")
        print(f"GPU VERIFICATION")
        print(f"{'='*50}")
        print(f"GPU Name: {gpu_info['name']}")
        print(f"Total VRAM: {gpu_info['total_vram_gb']:.1f} GB")
        print(f"Free VRAM: {gpu_info['free_vram_gb']:.1f} GB")
        print(f"Used VRAM: {gpu_info['used_vram_gb']:.1f} GB")
        print(f"Required VRAM: {self.MINIMUM_VRAM_GB:.1f} GB")
        print(f"{'='*50}")

        if gpu_info['total_vram_gb'] < self.MINIMUM_VRAM_GB:
            print(f"\n❌ GPU VERIFICATION FAILED")
            print(f"Your GPU has {gpu_info['total_vram_gb']:.1f} GB VRAM")
            print(f"Qwen3-VL-30B-A3B-Thinking-FP8 requires minimum {self.MINIMUM_VRAM_GB:.1f} GB VRAM")
            print(f"\nRecommended GPUs:")
            print(f"  - RTX 4090 (24GB)")
            print(f"  - RTX 6000 Ada (48GB)")
            print(f"  - A5000 (24GB)")
            print(f"  - A100 (40GB/80GB)")
            print(f"  - H100 (80GB)")
            return False

        print(f"\n✓ GPU VERIFICATION PASSED")
        print(f"Sufficient VRAM available for Qwen3-VL-30B-A3B-Thinking-FP8")

        return True

    def get_device_count(self) -> int:
        """Get the number of available GPU devices."""
        if not self.initialized:
            raise RuntimeError("GPU detector not initialized")

        try:
            return pynvml.nvmlDeviceGetCount()
        except Exception as e:
            print(f"ERROR: Failed to get device count: {e}")
            return 0

    def shutdown(self):
        """Shutdown NVML."""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except:
                pass

    def __del__(self):
        """Cleanup on destruction."""
        self.shutdown()


def verify_gpu_requirements(device_index: int = 0) -> bool:
    """
    Convenience function to verify GPU requirements.

    Args:
        device_index: GPU device index (default: 0)

    Returns:
        True if requirements are met, False otherwise
    """
    detector = GPUDetector()
    result = detector.verify_requirements(device_index)
    detector.shutdown()
    return result


if __name__ == "__main__":
    # Test the GPU detector
    print("Testing GPU Detection...")
    detector = GPUDetector()

    device_count = detector.get_device_count()
    print(f"\nDetected {device_count} GPU device(s)")

    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        info = detector.get_gpu_info(i)
        for key, value in info.items():
            if 'bytes' not in key:
                print(f"{key}: {value}")

        detector.verify_requirements(i)

    detector.shutdown()
