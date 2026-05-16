#!/usr/bin/env python3
# ruff: noqa: E402
import argparse
import csv
import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import pipeline as hf_pipeline

sys.stdout.reconfigure(encoding="utf-8")  # noqa


AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000
AST_DURATION = 10    # seconds to analyse per clip
THRESHOLD = 0.30  # music_score above this → "music"

MUSIC_KEYWORDS = {
    "music", "instrument", "singing", "choir", "piano", "guitar",
    "drum", "bass", "violin", "trumpet", "flute", "organ", "banjo",
    "ukulele", "harp", "cello", "synthesizer", "vocal", "song",
    "melody", "harmonic", "rhythm", "beat", "orchestra",
}

_YTDLP_CHAR_MAP = str.maketrans({
    ':': '：', '?': '？', '*': '＊', '"': '＂',
    '<': '＜', '>': '＞', '|': '｜', '\\': '⧹', '/': '⧸',
    '\t': ' ', '\n': ' ', '\r': ' ',
})

def sanitize_title(title: str) -> str:
    return title.translate(_YTDLP_CHAR_MAP).strip()[:100]


def find_mp4(clip: dict[str, str], directory: Path = Path(".")) -> Path | None:
    sanitized = sanitize_title(clip["title"])
    for path in directory.iterdir():
        if path.name.endswith(".mp4") and sanitized in path.name:
            return path
    return None


def classify_clip(filepath: Path, classifier: Any, duration: float,
                  threshold: float, debug: bool = False) -> tuple[str, float]:
    y, _ = librosa.load(filepath, sr=AST_SR, duration=duration, mono=True)
    results = classifier({"raw": y, "sampling_rate": AST_SR}, top_k=527)

    if debug:
        print("  All labels (score > 0.01):")
        for r in results:
            if r["score"] >= 0.01:
                print(f"    {r['score']:.4f}  {r['label']}")

    music_score = sum(
        r["score"] for r in results
        if any(kw in r["label"].lower() for kw in MUSIC_KEYWORDS)
    )
    label = "music" if music_score >= threshold else "not music"
    return label, round(music_score, 3)


def load_model(args: argparse.Namespace) -> Any:
    has_gpu = torch.cuda.is_available()
    if not has_gpu and not args.force_cpu:
        print("Error: no GPU detected. Use --force-cpu to run on CPU (will be slow).", file=sys.stderr)
        sys.exit(2)
    device = 0 if has_gpu else -1
    print(f"Loading model on {'GPU' if device == 0 else 'CPU'}...", file=sys.stderr)
    return hf_pipeline("audio-classification", model=AST_MODEL, device=device)


def mode_files(args: argparse.Namespace, classifier: Any) -> None:
    errors = 0
    for filepath in args.inputs:
        if not filepath.is_file():
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            errors += 1
            continue
        label, score = classify_clip(filepath, classifier, args.duration, args.threshold, args.debug)
        print(f"{filepath}\t{score:.3f}\t{label}")
    sys.exit(0 if errors == 0 else 2)


def mode_csv(csv_file: Path, args: argparse.Namespace, classifier: Any) -> None:
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.")
        return

    fieldnames = list(rows[0].keys())
    if "classification" not in fieldnames:
        fieldnames.append("classification")
        for row in rows:
            row["classification"] = ""

    pending = [r for r in rows if not r.get("classification")]
    print(
        f"Total rows: {len(rows)}, already classified: {len(rows) - len(pending)}, pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    def save():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    classified = skipped = 0
    for i, row in enumerate(pending, 1):
        mp4 = find_mp4(row)
        if mp4 is None:
            print(f"[{i:3d}/{len(pending)}] [no file ]  {row['title']}")
            skipped += 1
            continue

        label, score = classify_clip(mp4, classifier, args.duration, args.threshold, args.debug)
        row["classification"] = label
        classified += 1
        print(f"[{i:3d}/{len(pending)}] [{label:9s}]  score={score:.3f}  {row['title']}")

        if classified % 200 == 0:
            save()

    save()
    print(f"\nDone. Classified: {classified}, skipped (no file): {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify audio as music or not-music using the AST model."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more .mp4 files, or a single .csv file for batch mode.",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Music score threshold (default: {THRESHOLD}). "
        "Scores above this are classified as music.",
    )
    parser.add_argument(
        "--duration", type=float, default=AST_DURATION,
        help=f"Seconds of audio to analyse per clip (default: {AST_DURATION}).",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Allow running on CPU even without a GPU (slow).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print all detected labels and their scores for each clip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = load_model(args)

    if len(args.inputs) == 1 and args.inputs[0].suffix == ".csv":
        mode_csv(args.inputs[0], args, classifier)
    else:
        mode_files(args, classifier)


if __name__ == "__main__":
    main()
