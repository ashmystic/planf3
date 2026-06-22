#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.50.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Generate images with Google Gemini (default) or OpenAI's gpt-image-2.

Provider is selected with --provider (default: gemini). Gemini talks to the raw
generateContent REST endpoint (no extra deps); OpenAI uses the openai SDK.

Usage:
    python generate_gpt_image.py "prompt" output.png [options]

Examples:
    python generate_gpt_image.py "A sunset over mountains" sunset.png
    python generate_gpt_image.py "Company logo" logo.png --provider openai --quality high
    python generate_gpt_image.py "Wide cinematic shot" wide.png --size 1536x1024

Environment:
    GEMINI_API_KEY - Required when --provider gemini (default)
    OPENAI_API_KEY - Required when --provider openai
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


VALID_PROVIDERS = ["gemini", "openai"]
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash"
DEFAULT_OPENAI_MODEL = "gpt-image-2"

VALID_QUALITY = ["auto", "low", "medium", "high"]
VALID_FORMATS = ["png", "jpeg", "webp"]
VALID_MODERATION = ["auto", "low"]
# gpt-image-2 does NOT support "transparent" — only opaque/auto.
VALID_BACKGROUND = ["auto", "opaque"]

# Popular sizes — gpt-image-2 also accepts any custom size meeting:
#   max edge ≤ 3840, both edges multiples of 16, aspect ≤ 3:1, 655360–8294400 total px
POPULAR_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "3840x2160",
    "2160x3840",
]

# Aspect ratios Gemini's image models accept. We snap an OpenAI-style WxH --size to the
# nearest of these so the same --size flag works across both providers.
GEMINI_ASPECT_RATIOS = {
    "1:1": 1 / 1,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def size_to_aspect_ratio(size: str) -> str | None:
    """Map an OpenAI-style 'WxH' size to the nearest Gemini aspect ratio, or None for 'auto'."""
    if not size or size == "auto" or "x" not in size:
        return None
    try:
        w, h = (int(v) for v in size.lower().split("x", 1))
        ratio = w / h
    except (ValueError, ZeroDivisionError):
        return None
    return min(GEMINI_ASPECT_RATIOS, key=lambda k: abs(GEMINI_ASPECT_RATIOS[k] - ratio))


def gemini_generate_content(api_key: str, model: str, parts: list[dict], aspect_ratio: str | None):
    """POST to the Gemini generateContent REST endpoint; return (b64_data, mime_type)."""
    url = f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}"
    generation_config: dict = {"responseModalities": ["IMAGE"]}
    if aspect_ratio:
        generation_config["imageConfig"] = {"aspectRatio": aspect_ratio}
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        hint = " (check --model is an image-capable Gemini model)" if e.code in (400, 404) else ""
        raise RuntimeError(f"Gemini HTTP {e.code}: {detail[:300]}{hint}")

    candidates = payload.get("candidates", [])
    for cand in candidates:
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return inline["data"], mime
    finish = candidates[0].get("finishReason", "?") if candidates else "no candidates"
    raise RuntimeError(f"Gemini returned no image (finishReason={finish})")


def generate_gemini_image(
    prompt: str,
    output_path: str,
    model: str,
    size: str,
    n: int,
) -> None:
    """Generate one or more images using a Gemini image model via REST."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")

    aspect_ratio = size_to_aspect_ratio(size)

    print("Provider:   gemini")
    print(f"Model:      {model}")
    print(f"Aspect:     {aspect_ratio or 'default'} (from size {size})")
    print(f"Count:      {n}")
    print(f"Prompt:     {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print()
    print("Generating image...")

    out = Path(output_path)
    for i in range(n):
        b64_data, _mime = gemini_generate_content(api_key, model, [{"text": prompt}], aspect_ratio)
        target = out if n == 1 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        target.write_bytes(base64.b64decode(b64_data))
        print(f"Saved: {target}")


def generate_gpt_image(
    prompt: str,
    output_path: str,
    model: str = DEFAULT_OPENAI_MODEL,
    size: str = "auto",
    quality: str = "auto",
    n: int = 1,
    output_format: str = "png",
    output_compression: int | None = None,
    moderation: str = "auto",
    background: str = "auto",
) -> None:
    """Generate one or more images using gpt-image-2."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": n,
        "output_format": output_format,
        "moderation": moderation,
        "background": background,
    }
    if output_compression is not None and output_format in {"jpeg", "webp"}:
        kwargs["output_compression"] = output_compression

    print("Provider:   openai")
    print(f"Model:      {model}")
    print(f"Size:       {size}")
    print(f"Quality:    {quality}")
    print(f"Format:     {output_format}")
    print(f"Background: {background}")
    print(f"Count:      {n}")
    print(f"Prompt:     {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print()
    print("Generating image...")

    result = client.images.generate(**kwargs)

    out = Path(output_path)
    for i, item in enumerate(result.data):
        if n == 1:
            target = out
        else:
            target = out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        target.write_bytes(base64.b64decode(item.b64_json))
        print(f"Saved: {target}")

    if getattr(result, "usage", None):
        print(f"Usage: {result.usage}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Gemini (default) or OpenAI gpt-image-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", help="Text prompt describing the image")
    parser.add_argument("output", help="Output file path (e.g., output.png)")
    parser.add_argument(
        "--provider",
        "-p",
        default="gemini",
        choices=VALID_PROVIDERS,
        help="Image provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Model ID. Defaults per provider: "
            f"'{DEFAULT_GEMINI_MODEL}' (gemini), '{DEFAULT_OPENAI_MODEL}' (openai)."
        ),
    )
    parser.add_argument(
        "--size",
        "-s",
        default="auto",
        help=(
            "Image size WxH (default: auto). gpt-image-2 popular: "
            + ", ".join(POPULAR_SIZES)
            + ". For gemini the WxH is snapped to the nearest supported aspect ratio."
        ),
    )
    parser.add_argument(
        "--quality",
        "-q",
        default="auto",
        choices=VALID_QUALITY,
        help="Quality tier (openai only; default: auto)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of images to generate (default: 1; suffixes _1, _2, ... when >1)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="png",
        choices=VALID_FORMATS,
        help="Output format (openai only; default: png)",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=None,
        help="Output compression 0-100 (openai jpeg/webp only)",
    )
    parser.add_argument(
        "--moderation",
        default="auto",
        choices=VALID_MODERATION,
        help="Moderation strictness (openai only; default: auto)",
    )
    parser.add_argument(
        "--background",
        default="auto",
        choices=VALID_BACKGROUND,
        help=(
            "Background mode (openai only; default: auto). gpt-image-2 supports only "
            "'auto' or 'opaque' — 'transparent' is NOT supported by this model."
        ),
    )

    args = parser.parse_args()

    try:
        if args.provider == "gemini":
            generate_gemini_image(
                prompt=args.prompt,
                output_path=args.output,
                model=args.model or DEFAULT_GEMINI_MODEL,
                size=args.size,
                n=args.count,
            )
        else:
            generate_gpt_image(
                prompt=args.prompt,
                output_path=args.output,
                model=args.model or DEFAULT_OPENAI_MODEL,
                size=args.size,
                quality=args.quality,
                n=args.count,
                output_format=args.format,
                output_compression=args.compression,
                moderation=args.moderation,
                background=args.background,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
