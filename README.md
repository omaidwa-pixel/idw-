# Local Stable Diffusion CLI Image Generator

Generate images from text prompts locally using Hugging Face Diffusers and Stable Diffusion.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 generate_image.py "a futuristic city at sunset"
```

Optional arguments:

- `--model` (default: `runwayml/stable-diffusion-v1-5`)
- `--height` (default: `512`)
- `--width` (default: `512`)
- `--steps` (default: `30`)
- `--guidance-scale` (default: `7.5`)
- `--seed` (optional integer for reproducible images)

## Notes

- The script runs from the command line only (no hosted API, no website, no domain).
- Image generation requires dependencies in `requirements.txt` to be installed first.
- Generated images are saved under `outputs/images`.
