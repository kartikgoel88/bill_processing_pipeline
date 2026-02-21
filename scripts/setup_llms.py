#!/usr/bin/env python3
"""
One-time setup for LLMs and vision/document models used by the bill processing pipeline.

- Ollama (local): Llama, Qwen, LLaVA — pulls models via `ollama pull`
- Donut: HuggingFace receipt model — installs [document] and optionally pre-downloads
- LayoutLM: HuggingFace layout model — installs deps; extractor not yet implemented in pipeline

Usage:
  python scripts/setup_llms.py              # interactive: offer each group
  python scripts/setup_llms.py --all         # setup everything
  python scripts/setup_llms.py --ollama     # only Ollama models
  python scripts/setup_llms.py --ollama-model qwen2.5:7b   # pull a single Ollama model
  python scripts/setup_llms.py --donut      # only Donut (pip + optional download)
  python scripts/setup_llms.py --layoutlm   # only LayoutLM deps (pip)
  python scripts/setup_llms.py --llava       # only LLaVA (Ollama vision)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Ollama model tags used by the pipeline (see .env LLM_DECISION_MODEL, LLM_VISION_MODEL)
OLLAMA_DECISION_MODELS = [
    "llama3.2",
    "qwen2.5:7b",
    "qwen3:1.7b",
    "qwen3:4b",
]
OLLAMA_VISION_MODELS = [
    "llava",
]

# HuggingFace model IDs (see DONUT_MODEL_ID in .env)
DONUT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
LAYOUTLM_MODEL_ID = "microsoft/layoutlmv3-base"  # for when LayoutLM is implemented


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run command; return True if success."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd or PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"  Command not found: {cmd[0]}", file=sys.stderr)
        return False


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def pip_install_cmd() -> list[str]:
    """Return command to run for 'pip install' (prefer uv when venv has no pip)."""
    if shutil.which("uv") is not None:
        return ["uv", "pip", "install", "-e", ".[document]"]
    return [sys.executable, "-m", "pip", "install", "-e", ".[document]"]


def setup_ollama(*, decision: bool = True, vision: bool = True) -> bool:
    """Pull Ollama models for decision (Qwen, Llama) and/or vision (LLava)."""
    if not ollama_available():
        print("Ollama not found. Install from https://ollama.com and ensure `ollama` is on PATH.")
        return False
    models = []
    if decision:
        models.extend(OLLAMA_DECISION_MODELS)
    if vision:
        models.extend(OLLAMA_VISION_MODELS)
    ok = True
    for tag in models:
        print(f"  Pulling {tag} ...")
        if not run(["ollama", "pull", tag]):
            ok = False
    return ok


def setup_donut(*, install_deps: bool = True, download_model: bool = True) -> bool:
    """Install [document] extras and optionally pre-download Donut model."""
    if install_deps:
        print("  Installing [document] extras (transformers, torch) ...")
        if not run(pip_install_cmd(), cwd=PROJECT_ROOT):
            return False
    if download_model:
        print(f"  Pre-downloading Donut model {DONUT_MODEL_ID} ...")
        code = subprocess.call(
            [
                sys.executable,
                "-c",
                """
from transformers import VisionEncoderDecoderModel, AutoProcessor
model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
AutoProcessor.from_pretrained(model_id)
VisionEncoderDecoderModel.from_pretrained(model_id)
print("Done.")
""",
            ],
            cwd=PROJECT_ROOT,
        )
        if code != 0:
            print("  Donut download failed (model will load on first pipeline run).", file=sys.stderr)
            return False
    return True


def setup_layoutlm(*, install_deps: bool = True) -> bool:
    """Install deps for LayoutLM. Extractor not implemented in pipeline yet."""
    if install_deps:
        print("  Installing [document] extras (covers LayoutLM deps) ...")
        if not run(pip_install_cmd(), cwd=PROJECT_ROOT):
            return False
    print("  Note: Use VISION_EXTRACTOR=layoutlm and LAYOUTLM_MODEL_ID (e.g. nielsr/layoutlmv3-finetuned-cord) in .env to use LayoutLM.")
    return True


def setup_llava() -> bool:
    """Pull LLaVA via Ollama for vision_llm extractor."""
    return setup_ollama(decision=False, vision=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Setup LLMs and models for the bill processing pipeline")
    ap.add_argument("--all", action="store_true", help="Setup Ollama (decision+vision), Donut, LayoutLM deps")
    ap.add_argument("--ollama", action="store_true", help="Pull Ollama models: Llama, Qwen, LLaVA")
    ap.add_argument("--donut", action="store_true", help="Install Donut deps and pre-download model")
    ap.add_argument("--layoutlm", action="store_true", help="Install LayoutLM/document deps")
    ap.add_argument("--llava", action="store_true", help="Pull only LLaVA (Ollama vision)")
    ap.add_argument("--ollama-model", metavar="TAG", dest="ollama_model", help="Pull only this Ollama model (e.g. qwen2.5:7b, llama3.2)")
    ap.add_argument("--no-download-donut", action="store_true", help="With --donut: skip pre-download (load on first run)")
    args = ap.parse_args()

    if not any([args.all, args.ollama, args.donut, args.layoutlm, args.llava, args.ollama_model]):
        print("Choose one or more: --all, --ollama, --donut, --layoutlm, --llava, --ollama-model TAG")
        print("  Example: python scripts/setup_llms.py --all")
        print("  Example: python scripts/setup_llms.py --ollama-model qwen2.5:7b")
        return 0

    ok = True

    if args.ollama_model:
        if not ollama_available():
            print("Ollama not found. Install from https://ollama.com and ensure `ollama` is on PATH.")
            ok = False
        else:
            print(f"\n[Ollama] Pulling {args.ollama_model} ...")
            if not run(["ollama", "pull", args.ollama_model]):
                ok = False

    if args.all or args.ollama:
        print("\n[Ollama] Pulling decision + vision models (Llama, Qwen, LLaVA) ...")
        if not setup_ollama(decision=True, vision=True):
            ok = False

    if args.llava and not (args.all or args.ollama) and not args.ollama_model:
        print("\n[Ollama] Pulling LLaVA (vision) ...")
        if not setup_llava():
            ok = False

    if args.all or args.donut:
        print("\n[Donut] Installing and optionally downloading ...")
        if not setup_donut(download_model=not args.no_download_donut):
            ok = False

    if args.all or args.layoutlm:
        print("\n[LayoutLM] Installing document deps ...")
        if not setup_layoutlm():
            ok = False

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
