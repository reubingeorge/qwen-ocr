"""
Model Selector Module

Automatically selects the optimal Qwen3-VL model based on available VRAM.
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# Model configurations: (model_name, min_vram_gb, max_vram_gb)
# Qwen3-VL uses Mixture of Experts (MoE) architecture
MODEL_CONFIGS = [
    ("Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", 0, 24),      # FP8 quantized 30B model
    ("Qwen/Qwen3-VL-30B-A3B-Instruct", 24, 48),        # Full precision 30B model
    ("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 48, 80),  # FP8 quantized 235B model
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", 80, float('inf')),  # Full precision 235B model
]


class ModelSelector:
    """Selects appropriate Qwen3-VL model based on available VRAM."""

    def __init__(self):
        self.model_configs = MODEL_CONFIGS

    def select_model(self, vram_gb: float, compute_capability: float = 0.0, manual_model: str = None) -> str:
        """
        Select appropriate model based on VRAM and compute capability.

        Args:
            vram_gb: Available VRAM in GB
            compute_capability: GPU compute capability (e.g., 7.5, 8.9)
            manual_model: Manual model override (if provided)

        Returns:
            Model name string
        """
        # If manual model specified, validate and use it
        if manual_model and manual_model.lower() != 'auto':
            logger.info(f"Using manually specified model: {manual_model}")
            return manual_model

        # Auto-select based on VRAM
        if vram_gb == 0:
            logger.warning("No GPU detected. Using smallest model (CPU mode).")
            return self.model_configs[0][0]

        # Determine if GPU supports FP8 (compute capability >= 8.9)
        supports_fp8 = compute_capability >= 8.9

        for model_name, min_vram, max_vram in self.model_configs:
            # Skip FP8 models if GPU doesn't support them
            if "FP8" in model_name and not supports_fp8:
                logger.debug(f"Skipping FP8 model {model_name} - GPU compute capability {compute_capability} < 8.9")
                continue

            if min_vram <= vram_gb < max_vram:
                logger.info(f"Auto-selected model: {model_name} for {vram_gb:.1f}GB VRAM (Compute {compute_capability})")
                return model_name

        # Fallback to smallest non-FP8 model
        for model_name, _, _ in self.model_configs:
            if "FP8" not in model_name:
                fallback = model_name
                logger.warning(f"Could not determine optimal model. Using fallback: {fallback}")
                return fallback

        # Last resort - use first model
        fallback = self.model_configs[0][0]
        logger.warning(f"Using first available model: {fallback}")
        return fallback

    def get_model_info(self, vram_gb: float) -> Tuple[str, str]:
        """
        Get model name and description for given VRAM.

        Args:
            vram_gb: Available VRAM in GB

        Returns:
            Tuple of (model_name, description)
        """
        model_name = self.select_model(vram_gb)

        if "30B" in model_name and "FP8" in model_name:
            desc = "Quantized 30B MoE model - Good balance of speed and accuracy (FP8)"
        elif "30B" in model_name:
            desc = "Full precision 30B MoE model - High quality OCR"
        elif "235B" in model_name and "FP8" in model_name:
            desc = "Quantized 235B MoE model - Excellent accuracy (FP8)"
        elif "235B" in model_name:
            desc = "Full precision 235B MoE model - Best accuracy, requires large GPU"
        else:
            desc = "Custom model"

        return model_name, desc

    def list_available_models(self) -> Dict[str, Tuple[float, float]]:
        """
        List all available models with their VRAM requirements.

        Returns:
            Dict mapping model names to (min_vram, max_vram) tuples
        """
        return {
            model_name: (min_vram, max_vram)
            for model_name, min_vram, max_vram in self.model_configs
        }


def select_model_for_vram(vram_gb: float, compute_capability: float = 0.0, manual_model: str = None) -> str:
    """
    Convenience function to select model.

    Args:
        vram_gb: Available VRAM in GB
        compute_capability: GPU compute capability
        manual_model: Manual model override

    Returns:
        Model name string
    """
    selector = ModelSelector()
    return selector.select_model(vram_gb, compute_capability, manual_model)


if __name__ == "__main__":
    # Test model selection
    logging.basicConfig(level=logging.INFO)
    selector = ModelSelector()

    print("Model Selection Test")
    print("=" * 50)

    test_vrams = [0, 12, 24, 36, 48, 64, 80, 100]
    for vram in test_vrams:
        model, desc = selector.get_model_info(vram)
        print(f"VRAM: {vram:2d}GB → Model: {model}")
        print(f"          Description: {desc}")
        print()

    print("\nAvailable Models:")
    print("=" * 50)
    for model, (min_v, max_v) in selector.list_available_models().items():
        max_str = f"{max_v:.0f}GB" if max_v != float('inf') else "∞"
        print(f"{model}")
        print(f"  VRAM Range: {min_v:.0f}GB - {max_str}")
