#!/usr/bin/env python3
"""Local Stable Diffusion image generator (CLI only)."""

import argparse
import os
import re
import sys
from datetime import datetime


DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs/images"


def safe_filename(text: str, max_len: int = 80) -> str:
    """Convert prompt text to a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_")
    if not cleaned:
        cleaned = "image"
    return cleaned[:max_len]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image locally from a text prompt using Stable Diffusion."
    )
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels")
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of denoising inference steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output",
    )
    return parser


def generate_image(args: argparse.Namespace) -> str:
    """Generate one image and return the output path."""
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    pipeline = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipeline = pipeline.to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = pipeline(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = safe_filename(args.prompt)
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{name}.png")
    image.save(output_path)
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        output_path = generate_image(args)
        print(f"Saved image: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("Generation cancelled by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
