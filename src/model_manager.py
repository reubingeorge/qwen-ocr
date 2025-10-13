"""
Model Manager Module

This module handles loading and managing the Qwen3-VL-30B-A3B-Thinking-FP8 model
using vLLM for optimized inference.
"""

import sys
from typing import Optional
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: pip install vllm transformers")
    sys.exit(1)


class ModelManager:
    """Manages model loading and inference using vLLM."""

    MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"

    def __init__(self, gpu_memory_utilization: float = 0.70, tensor_parallel_size: int = 1):
        """
        Initialize the model manager.

        Args:
            gpu_memory_utilization: GPU memory utilization ratio (0.1-0.95)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = self.MODEL_NAME
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.llm: Optional[LLM] = None
        self.processor = None

    def load_model(self):
        """
        Load the Qwen3-VL-30B-A3B-Thinking-FP8 model using vLLM.

        This method loads both the model and the processor.
        """
        print(f"\nLoading model: {self.model_name}")
        print(f"GPU Memory Utilization: {self.gpu_memory_utilization}")
        print(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        print("This may take several minutes...")

        try:
            # Load processor
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("✓ Processor loaded")

            # Load model with vLLM
            print("Loading model with vLLM...")
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=4096,  # Reduced for better memory fit (was 8192)
                dtype="auto",  # Auto-detect dtype (FP8)
            )
            print("✓ Model loaded successfully")

        except Exception as e:
            print(f"\nERROR: Failed to load model: {e}")
            print("\nPossible solutions:")
            print("1. Check that you have sufficient VRAM (minimum 20GB)")
            print("2. Reduce gpu_memory_utilization (e.g., --gpu-memory 0.60)")
            print("3. Ensure vLLM is properly installed")
            print("4. Check that CUDA is available")
            sys.exit(1)

    def generate(self, messages: list, max_tokens: int = 4096, temperature: float = 0.1) -> str:
        """
        Generate text using the loaded model.

        Args:
            messages: List of message dictionaries (e.g., [{"role": "user", "content": [...]}])
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Set sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                stop=["<|im_end|>", "<|endoftext|>"],
            )

            # Generate
            outputs = self.llm.generate(prompt, sampling_params)

            # Extract text from output
            if outputs and len(outputs) > 0:
                return outputs[0].outputs[0].text.strip()
            else:
                return ""

        except Exception as e:
            print(f"ERROR: Generation failed: {e}")
            return ""

    def process_image_with_prompt(self, image_path: str, prompt: str,
                                   max_tokens: int = 4096, temperature: float = 0.1) -> str:
        """
        Process an image with a text prompt (OCR task).

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text (OCR result)
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Create message with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            return self.generate(messages, max_tokens, temperature)

        except Exception as e:
            print(f"ERROR: Image processing failed: {e}")
            return ""

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.llm is not None

    def unload(self):
        """Unload the model and free GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None

            # Try to clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("✓ GPU cache cleared")
            except:
                pass

            print("✓ Model unloaded")


def test_model():
    """Test the model loading and basic inference."""
    print("Testing Model Manager...")

    # Initialize model manager
    manager = ModelManager(gpu_memory_utilization=0.70)

    # Load model
    manager.load_model()

    # Test text generation
    print("\nTesting text generation...")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is OCR?"
                }
            ]
        }
    ]

    response = manager.generate(messages, max_tokens=100)
    print(f"Response: {response}")

    # Unload
    manager.unload()
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_model()
