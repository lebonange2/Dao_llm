#!/usr/bin/env python3
"""
Local data preparation script — run this on your machine before training.

Scrapes & sanitises Taoist texts from multiple sources, saves as a
ready-to-upload taoist_data.jsonl file.  Then upload it in the Gradio UI.

Usage:
    python scrape_data.py                              # ctext + sacred_texts
    python scrape_data.py --sources sacred_texts       # English only
    python scrape_data.py --sources ctext sacred_texts nanhuaijin
    python scrape_data.py --no-scrape                  # offline, seed corpus only
    python scrape_data.py --preview                    # print first 5 samples
    python scrape_data.py --output my_data.jsonl       # custom output path
"""

import re
import json
import logging
import argparse
from pathlib import Path

from sources import FALLBACK_PAIRS, SOURCE_MAP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Taoist instruction dataset locally, then upload via UI.")
    parser.add_argument("--output",    default="taoist_data.jsonl",
                        help="Output JSONL path (default: taoist_data.jsonl)")
    parser.add_argument("--sources",   nargs="+", default=["ctext", "sacred_texts"],
                        choices=list(SOURCE_MAP.keys()),
                        help="Sources to scrape (default: ctext sacred_texts)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip network calls — use built-in seed corpus only")
    parser.add_argument("--preview",   action="store_true",
                        help="Print first 5 samples after saving")
    args = parser.parse_args()

    print("=" * 65)
    print("  Taoist LLM — Local Dataset Builder")
    print("=" * 65)

    all_pairs = []

    if args.no_scrape:
        print("\n[1/3] --no-scrape set — using built-in seed corpus only.")
    else:
        print(f"\n[1/3] Scraping sources: {args.sources}")
        for src_name in args.sources:
            cls = SOURCE_MAP[src_name]
            src = cls()
            items = src.collect()
            all_pairs.extend(items)
            print(f"      {src_name}: {len(items)} passages")

    # Always include the seed corpus as a floor
    all_pairs.extend(list(FALLBACK_PAIRS))

    # Deduplicate
    seen: set = set()
    deduped = []
    for p in all_pairs:
        key = re.sub(r"\s+", "", p.get("instruction", "")).lower()[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    print(f"\n[2/3] {len(deduped)} unique instruction-response pairs assembled.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in deduped:
            record = {
                "instruction": p.get("instruction", ""),
                "response":    p.get("response", ""),
                "source":      p.get("source", "fallback"),
                "language":    p.get("language", "english"),
                "text":        f"### Instruction:\n{p.get('instruction','')}\n\n### Response:\n{p.get('response','')}",
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    size_kb = out_path.stat().st_size // 1024
    print(f"\n[3/3] Saved → {out_path.resolve()}  ({size_kb} KB)")
    print(f"\n✅  Done!  Upload '{out_path.name}' in the Gradio UI before starting training.")

    if args.preview:
        print("\n── First 5 samples ──────────────────────────────────────────")
        for i, line in enumerate(out_path.read_text(encoding="utf-8").splitlines()[:5], 1):
            rec = json.loads(line)
            print(f"\n  [{i}] INSTRUCTION: {rec['instruction'][:100]}")
            print(f"       RESPONSE:    {rec['response'][:100]}…")
    print("=" * 65)


if __name__ == "__main__":
    main()
