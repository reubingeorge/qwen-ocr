"""
Model Manager Module

Handles loading, caching, and managing Qwen3-VL models.
"""

import logging
from typing import Optional, Tuple, Any
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages Qwen3-VL MoE model loading and caching."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize model manager.

        Args:
            model_name: Name of the Qwen model to load
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self) -> bool:
        """
        Load the model and processor.

        Returns:
            True if successful, False otherwise
        """
        if self._loaded:
            logger.info("Model already loaded")
            return True

        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Device: {self.device}")

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            logger.info("Loading model (this may take a few minutes)...")
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"✓ Model loaded successfully: {self.model_name}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            self._loaded = False
            return False

    def generate_text(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> Optional[str]:
        """
        Generate text from image using the VLM.

        Args:
            image_path: Path to image file
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text or None if error
        """
        if not self._loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            # Create message with image and prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Prepare inputs using apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=False if temperature == 0 else True,
                )

            # Trim input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._loaded:
            logger.info("Unloading model from memory...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dict with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._loaded,
        }


def create_model_manager(model_name: str, device: str = "cuda") -> Optional[ModelManager]:
    """
    Convenience function to create and load a model manager.

    Args:
        model_name: Name of the model
        device: Device to use

    Returns:
        Loaded ModelManager instance or None if failed
    """
    manager = ModelManager(model_name, device)
    if manager.load_model():
        return manager
    return None


if __name__ == "__main__":
    # Test model manager
    logging.basicConfig(level=logging.INFO)

    print("Model Manager Test")
    print("=" * 80)

    # Test with smallest model
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing with: {model_name}")
    print(f"Device: {device}")
    print()

    manager = ModelManager(model_name, device)

    print("Loading model...")
    success = manager.load_model()

    if success:
        print("✓ Model loaded successfully")
        print()
        print("Model info:")
        for key, value in manager.get_model_info().items():
            print(f"  {key}: {value}")

        print("\nUnloading model...")
        manager.unload_model()
        print("✓ Model unloaded")
    else:
        print("✗ Failed to load model")
