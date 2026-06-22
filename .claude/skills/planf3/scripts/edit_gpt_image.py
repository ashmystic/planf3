#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.50.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Edit/compose existing images with Google Gemini (default) or OpenAI's gpt-image-2.

Provider is selected with --provider (default: gemini). Gemini talks to the raw
generateContent REST endpoint (no extra deps); OpenAI uses the openai SDK.
Pass one or more input images. With multiple inputs the model composes them.

Usage:
    python edit_gpt_image.py "edit instruction" output.png input.png [more.png ...] [options]

Examples:
    python edit_gpt_image.py "Add a rainbow in the sky" edited.png photo.png
    python edit_gpt_image.py "Make a group photo" group.png p1.png p2.png p3.png
    python edit_gpt_image.py "Add a rainbow" edited.png photo.png --provider openai

Environment:
    GEMINI_API_KEY - Required when --provider gemini (default)
    OPENAI_API_KEY - Required when --provider openai
"""

import argparse
import base64
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


VALID_PROVIDERS = ["gemini", "openai"]
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash"
DEFAULT_OPENAI_MODEL = "gpt-image-2"

VALID_QUALITY = ["auto", "low", "medium", "high"]
VALID_FORMATS = ["png", "jpeg", "webp"]
# gpt-image-2 does NOT support "transparent" — only opaque/auto.
VALID_BACKGROUND = ["auto", "opaque"]

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def backup_if_exists(output_path: str) -> None:
    """Copy an existing output file into ./backup/ before it gets overwritten.

    Edits often target a path that already holds an image (sometimes the input
    itself), so back the original up first — losing it to an edit is silent and
    unrecoverable. backup/ self-ignores via a backup/.gitignore of "*".
    """
    out = Path(output_path)
    if not out.exists():
        return
    backup_dir = Path.cwd() / "backup"
    backup_dir.mkdir(exist_ok=True)
    gitignore = backup_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = backup_dir / f"{out.stem}_{ts}{out.suffix}"
    counter = 1
    while dest.exists():
        dest = backup_dir / f"{out.stem}_{ts}_{counter}{out.suffix}"
        counter += 1
    shutil.copy2(out, dest)
    print(f"Backed up existing {output_path} -> {dest}")


def mime_for(path: str) -> str:
    return MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "image/png")


def gemini_generate_content(api_key: str, model: str, parts: list[dict]):
    """POST to the Gemini generateContent REST endpoint; return (b64_data, mime_type)."""
    url = f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

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


def edit_gemini_image(
    input_paths: list[str],
    instruction: str,
    output_path: str,
    model: str,
) -> None:
    """Edit/compose images using a Gemini image model via REST."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")

    for p in input_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input image not found: {p}")

    # Input images first, then the text instruction (mirrors Gemini's image-edit examples).
    parts: list[dict] = []
    for p in input_paths:
        data = base64.b64encode(Path(p).read_bytes()).decode("ascii")
        parts.append({"inline_data": {"mime_type": mime_for(p), "data": data}})
    parts.append({"text": instruction})

    print("Provider:   gemini")
    print(f"Model:      {model}")
    print(f"Inputs:     {', '.join(input_paths)}")
    print(f"Prompt:     {instruction[:120]}{'...' if len(instruction) > 120 else ''}")
    print()
    print("Editing image...")

    b64_data, _mime = gemini_generate_content(api_key, model, parts)
    backup_if_exists(output_path)
    Path(output_path).write_bytes(base64.b64decode(b64_data))
    print(f"Saved: {output_path}")


def edit_gpt_image(
    input_paths: list[str],
    instruction: str,
    output_path: str,
    model: str = DEFAULT_OPENAI_MODEL,
    size: str = "auto",
    quality: str = "auto",
    output_format: str = "png",
    output_compression: int | None = None,
    mask_path: str | None = None,
    background: str = "auto",
) -> None:
    """Edit/compose images using gpt-image-2."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    for p in input_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input image not found: {p}")

    client = OpenAI(api_key=api_key)

    image_files = [open(p, "rb") for p in input_paths]
    try:
        kwargs = {
            "model": model,
            "image": image_files if len(image_files) > 1 else image_files[0],
            "prompt": instruction,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
        }
        if output_compression is not None and output_format in {"jpeg", "webp"}:
            kwargs["output_compression"] = output_compression
        if mask_path:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            kwargs["mask"] = open(mask_path, "rb")

        print("Provider:   openai")
        print(f"Model:      {model}")
        print(f"Inputs:     {', '.join(input_paths)}")
        print(f"Size:       {size}")
        print(f"Quality:    {quality}")
        print(f"Format:     {output_format}")
        print(f"Background: {background}")
        print(f"Prompt:     {instruction[:120]}{'...' if len(instruction) > 120 else ''}")
        print()
        print("Editing image...")

        result = client.images.edit(**kwargs)
    finally:
        for f in image_files:
            f.close()
        if mask_path and "mask" in kwargs:
            kwargs["mask"].close()

    item = result.data[0]
    backup_if_exists(output_path)
    Path(output_path).write_bytes(base64.b64decode(item.b64_json))
    print(f"Saved: {output_path}")

    if getattr(result, "usage", None):
        print(f"Usage: {result.usage}")


def main():
    parser = argparse.ArgumentParser(
        description="Edit/compose images with Gemini (default) or OpenAI gpt-image-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("instruction", help="Edit/compose instruction")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input image paths (multiple = composition)",
    )
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
        help="Image size WxH (openai only; default: auto). E.g. 1024x1024, 1536x1024.",
    )
    parser.add_argument(
        "--quality",
        "-q",
        default="auto",
        choices=VALID_QUALITY,
        help="Quality tier (openai only; default: auto)",
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
        "--mask",
        default=None,
        help="Optional mask PNG (openai only; transparent areas = regions to edit)",
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
            edit_gemini_image(
                input_paths=args.inputs,
                instruction=args.instruction,
                output_path=args.output,
                model=args.model or DEFAULT_GEMINI_MODEL,
            )
        else:
            edit_gpt_image(
                input_paths=args.inputs,
                instruction=args.instruction,
                output_path=args.output,
                model=args.model or DEFAULT_OPENAI_MODEL,
                size=args.size,
                quality=args.quality,
                output_format=args.format,
                output_compression=args.compression,
                mask_path=args.mask,
                background=args.background,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
