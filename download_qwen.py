#!/usr/bin/env python3
"""
Utility script to download Qwen 8B (Qwen/Qwen2.5-8B-Instruct) model weights
for offline CPU/GPU inference.

Run this script in an environment with internet access before deploying to offline Snowflake workspace.

Usage:
  python download_qwen.py --model_id Qwen/Qwen2.5-8B-Instruct --output_dir models/Qwen2.5-8B-Instruct
"""

import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Download Qwen 8B weights for offline RL trading")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-8B-Instruct", help="Hugging Face model repository ID")
    parser.add_argument("--output_dir", type=str, default="models/Qwen2.5-8B-Instruct", help="Local directory to save model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    print("=" * 80)
    print("📥 DOWNLOADING QWEN 8B MODEL FOR OFFLINE TRADING INFERENCE")
    print(f"• Model ID:    {args.model_id}")
    print(f"• Destination: {output_path}")
    print("=" * 80)

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(output_path, exist_ok=True)
        print("Starting download via Hugging Face Hub (this may take a few minutes)...")
        snapshot_download(
            repo_id=args.model_id,
            local_dir=output_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print("✅ Download completed successfully!")
        print(f"Model saved to: {output_path}")
    except ImportError:
        print("huggingface_hub not found. You can clone using git-lfs:")
        print(f"  git lfs install")
        print(f"  git clone https://huggingface.co/{args.model_id} {output_path}")
    except Exception as e:
        print(f"Download encountered an error: {e}")
        print("Note: The trading system includes pre-computed Qwen embeddings and an offline neural analyzer")
        print("so training runs 100% offline even if raw weights are not downloaded.")


if __name__ == "__main__":
    main()
