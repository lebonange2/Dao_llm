#!/usr/bin/env python3
"""
Standalone data preparation script — run this LOCALLY before training.

Usage:
    python scrape_data.py                        # scrape + save to ./taoist_data.jsonl
    python scrape_data.py --output my_data.jsonl # custom output path
    python scrape_data.py --preview              # print first 10 samples after saving

Then upload the resulting .jsonl file in the Gradio UI before starting training.
"""

import re
import json
import time
import argparse
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Sources ──────────────────────────────────────────────────────────────────

SOURCES = [
    {
        "name": "Dao De Jing (道德經)",
        "url":  "https://ctext.org/dao-de-jing",
    },
    {
        "name": "Zhuangzi (莊子)",
        "url":  "https://ctext.org/zhuangzi",
    },
    {
        "name": "Liezi (列子)",
        "url":  "https://ctext.org/liezi",
    },
    {
        "name": "Huainanzi (淮南子)",
        "url":  "https://ctext.org/huainanzi",
    },
]

# ── Built-in seed corpus (always included for robustness) ────────────────────

SEED_CORPUS = [
    "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。",
    "上善若水。水善利万物而不争，处众人之所恶，故几于道。",
    "天下皆知美之为美，斯恶已。皆知善之为善，斯不善已。",
    "为学日益，为道日损。损之又损，以至於无为。无为而无不为。",
    "知人者智，自知者明。胜人者有力，自胜者强。",
    "致虚极，守静笃。万物并作，吾以观复。",
    "信言不美，美言不信。善者不辩，辩者不善。",
    "曲则全，枉则直，洼则盈，弊则新，少则得，多则惑。",
    "知常容，容乃公，公乃全，全乃天，天乃道，道乃久，没身不殆。",
    "合抱之木，生於毫末；九层之台，起於累土；千里之行，始於足下。",
    "天下莫柔弱於水，而攻坚强者莫之能胜，以其无以易之。",
    "道生一，一生二，二生三，三生万物。万物负阴而抱阳，冲气以为和。",
    "祸兮福之所倚，福兮祸之所伏。孰知其极？其无正也。",
    "江海之所以能为百谷王者，以其善下之，故能为百谷王。",
    "天之道，利而不害；圣人之道，为而不争。",
    "知足不辱，知止不殆，可以长久。",
    "我有三宝，持而保之：一曰慈，二曰俭，三曰不敢为天下先。",
    "圣人不积，既以为人己愈有，既以与人己愈多。",
    "归根曰静，是谓复命。复命曰常，知常曰明。",
    "知者不言，言者不知。塞其兑，闭其门，挫其锐，解其纷。",
]


# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_source(url: str, name: str, max_paragraphs: int = 500) -> list[str]:
    headers = {"User-Agent": "TaoistLLM-Research/1.0 (Educational/Non-commercial)"}
    texts = []
    try:
        print(f"  Fetching {name} …", end=" ", flush=True)
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for p in soup.select("div.ctext, p.chinese, .passage, td.ctext"):
            text = p.get_text(strip=True)
            if _is_valid_chinese(text):
                texts.append(text)
                if len(texts) >= max_paragraphs:
                    break
        print(f"got {len(texts)} passages.")
    except Exception as exc:
        print(f"FAILED ({exc})")
    time.sleep(1.0)
    return texts


# ── Sanitisation ─────────────────────────────────────────────────────────────

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ALLOWED_RE = re.compile(
    r"^[\u4e00-\u9fff\u3400-\u4dbf\s，。？！；：、""''（）《》【】\-—…·○●◎]+$"
)


def _is_valid_chinese(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    chinese_chars = len(_CHINESE_RE.findall(text))
    if chinese_chars / max(len(text), 1) < 0.5:
        return False
    return bool(_ALLOWED_RE.match(text))


def sanitise(passages: list[str]) -> list[str]:
    clean = []
    seen = set()
    for raw in passages:
        text = unicodedata.normalize("NFC", raw).strip()
        text = re.sub(r"\s+", " ", text)
        if not _is_valid_chinese(text):
            continue
        key = re.sub(r"\s", "", text)
        if key in seen:
            continue
        seen.add(key)
        clean.append(text)
    return clean


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape & sanitise Taoist corpus locally.")
    parser.add_argument("--output",  default="taoist_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--preview", action="store_true",         help="Print first 10 samples after saving")
    parser.add_argument("--max",     type=int, default=500,       help="Max passages per source")
    args = parser.parse_args()

    print("=" * 60)
    print("  Taoist Corpus — Local Scraper & Sanitiser")
    print("=" * 60)

    # 1. Scrape
    raw = list(SEED_CORPUS)
    print("\n[1/3] Scraping sources …")
    for src in SOURCES:
        raw.extend(scrape_source(src["url"], src["name"], max_paragraphs=args.max))

    # 2. Sanitise
    print(f"\n[2/3] Sanitising {len(raw)} raw passages …")
    clean = sanitise(raw)
    print(f"      {len(clean)} passages retained after dedup + quality filter.")

    if len(clean) < 20:
        print(f"\n⚠️  Only {len(clean)} passages collected.")
        print("   The scrape may have been blocked. The seed corpus is still included.")
        print("   You can manually add more passages to the output JSONL file.")

    # 3. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in clean:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n[3/3] Saved {len(clean)} passages → {out_path.resolve()}")
    print(f"\n✅  Done!  Upload '{out_path.name}' in the Gradio UI before starting training.")

    if args.preview:
        print("\n── First 10 samples ──────────────────────────────────────")
        for i, line in enumerate(out_path.read_text(encoding="utf-8").splitlines()[:10], 1):
            print(f"  {i:2d}. {json.loads(line)['text'][:80]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
