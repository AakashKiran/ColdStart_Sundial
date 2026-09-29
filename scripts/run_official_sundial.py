"""
Run the official/reference Sundial implementation.

In this project, "official" refers to the released Sundial implementation
stored under:

    reference/sundial-base-128m/

This script loads the model and checkpoint directly from that reference
directory and does not use the implementation under
src/original_sundial_replication/.

The purpose of this script is to establish a working baseline for the
released Sundial implementation before validating our replication.

Some Definitions
    1. lookback_length = 2880
        This is the amount of past context given to Sundial.
    2. forecast_length = 96
        This is how many future time points we ask Sundial to generate.
    3. num_samples = 20
        Sundial is generative, so instead of asking for one possible future, we're asking for 20 forecast samples.
"""

from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "reference" / "sundial-base-128m"


def main():
    print(f"Loading Sundial from: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the released Sundial implementation and its checkpoint.
    # trust_remote_code=True allows Transformers to load Sundial's
    # custom model implementation from the reference directory.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Model class: {model.__class__.__name__}")

    # Use the context length and forecast configuration from the
    # official Sundial usage example.
    lookback_length = 2880
    forecast_length = 96
    num_samples = 20

    # Fixed seed so that this baseline run can be reproduced.
    torch.manual_seed(42)

    # Create a synthetic time series only for execution testing.
    # This is NOT a real dataset and has no forecasting meaning.
    sequences = torch.randn(
        1,
        lookback_length,
        device=device,
    )

    print(f"Input shape: {sequences.shape}")
    print(f"Forecast length: {forecast_length}")
    print(f"Number of samples: {num_samples}")

    print("Running official Sundial inference...")

    start_time = time.perf_counter()

    with torch.inference_mode():
        forecasts = model.generate(
            sequences,
            max_new_tokens=forecast_length,
            num_samples=num_samples,
        )

    elapsed = time.perf_counter() - start_time

    print("\nInference completed.")
    print(f"Output type: {type(forecasts)}")

    if hasattr(forecasts, "shape"):
        print(f"Output shape: {forecasts.shape}")

    # Report basic properties and numerical sanity checks for the
    # generated forecast tensor.
    if isinstance(forecasts, torch.Tensor):
        print(f"Output dtype: {forecasts.dtype}")
        print(f"Output device: {forecasts.device}")
        print(f"All values finite: {torch.isfinite(forecasts).all().item()}")
        print(f"Output mean: {forecasts.mean().item():.6f}")
        print(f"Output std: {forecasts.std().item():.6f}")

    print(f"Inference time: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()