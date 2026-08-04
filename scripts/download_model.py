#!/usr/bin/env python3
"""
Downloads a Hugging Face model to a local directory for offline use.
Defaults to 'google/gemma-3-1b-it', which can be overridden via command-line arguments.

Usage:
    python scripts/download_model.py [model_id] [output_dir]

Example:
    python scripts/download_model.py google/gemma-3-1b-it models/gemma-3-1b-it
"""
import argparse
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download HF model for offline use.")
    parser.add_argument("model_id", nargs="?", default="google/gemma-3-1b-it", help="Hugging Face model ID")
    parser.add_argument("output_dir", nargs="?", default="models/gemma-3-1b-it", help="Local directory to save the model")
    args = parser.parse_args()

    print(f"Downloading model '{args.model_id}' to '{args.output_dir}'...")
    
    try:
        snapshot_download(
            repo_id=args.model_id,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,  # Ensure actual files are present for portability
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.msgpack"], # prefer safetensors/pytorch
        )
        print(f"
Success! Model downloaded to: {Path(args.output_dir).resolve()}")
        print(f"
To use this model offline with the dualstream CLI:")
        print(f"  python -m dualstream.cli generate --model {args.output_dir} --prompt "..."")
    except Exception as e:
        print(f"
Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
