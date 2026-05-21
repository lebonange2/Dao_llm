#!/usr/bin/env python3
"""
Standalone data preparation script — run this LOCALLY before training.

Produces an English-language instruction-response dataset based on Taoist
philosophy (Tao Te Ching, Zhuangzi, Liezi).  Run it once on your machine,
review the output .jsonl, then upload it via the Gradio UI.

Usage:
    python scrape_data.py                        # generate taoist_data.jsonl
    python scrape_data.py --output my_data.jsonl # custom output path
    python scrape_data.py --preview              # print first 5 samples
    python scrape_data.py --no-scrape            # use seed corpus only (offline)
"""

import re
import json
import time
import argparse
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# SEED CORPUS — English instruction / response pairs (public domain content)
# These are always included regardless of scraping success.
# ─────────────────────────────────────────────────────────────────────────────

SEED_PAIRS = [
    {
        "instruction": "What is the Tao?",
        "response": (
            "The Tao (道) is the fundamental principle underlying all of reality. "
            "It cannot be fully captured in words — 'The Tao that can be told is not the eternal Tao.' "
            "It is the source from which all things arise, the nameless beginning of heaven and earth. "
            "Rather than a deity or a force to be commanded, it is the natural way of the universe itself: "
            "silent, effortless, and inexhaustible."
        ),
    },
    {
        "instruction": "What does the Tao Te Ching teach about water?",
        "response": (
            "Water is one of the central metaphors in the Tao Te Ching. "
            "'The highest good is like water' (上善若水). Water benefits all living things without competing, "
            "dwelling willingly in the lowest places that others disdain. "
            "This teaches the Taoist virtues of humility, non-contention, and yielding strength: "
            "nothing in the world is softer than water, yet nothing surpasses it in overcoming the hard and strong."
        ),
    },
    {
        "instruction": "What is wu wei?",
        "response": (
            "Wu wei (無為) literally means 'non-doing' or 'effortless action.' "
            "It does not mean laziness or passivity, but rather acting in complete harmony with the natural flow of things — "
            "without force, without ego, without unnecessary interference. "
            "The sage accomplishes everything through wu wei: 'Do nothing, and nothing is left undone.' "
            "Like a skilled craftsman whose art appears effortless, or water carving stone without straining, "
            "wu wei is action that arises naturally from one's deepest nature."
        ),
    },
    {
        "instruction": "How should a leader govern according to Taoist philosophy?",
        "response": (
            "The Tao Te Ching describes the ideal ruler as one who governs through stillness and restraint. "
            "'Governing a large state is like cooking a small fish' — handle it too much and you ruin it. "
            "The best leader is barely noticed by the people. When the work is done and the goal achieved, "
            "the people say: 'We did it ourselves.' "
            "By not imposing their will, not interfering unnecessarily, and embodying simplicity, "
            "such leaders create the conditions for natural harmony and flourishing."
        ),
    },
    {
        "instruction": "What does Taoism say about the relationship between opposites?",
        "response": (
            "Taoism sees opposites not as contradictions but as complementary aspects of a single whole. "
            "Beauty only exists because we recognise ugliness; good only because we know bad. "
            "Hard and soft, long and short, high and low — each gives meaning to the other. "
            "This interdependence is captured in the image of yin and yang: two forces that generate "
            "and balance each other in endless, dynamic harmony. "
            "The sage, understanding this, does not cling to one side or resist the other."
        ),
    },
    {
        "instruction": "What does the Tao Te Ching mean by 'returning to the root'?",
        "response": (
            "'Returning to the root' (歸根) means coming back to stillness, to the source, to one's original nature. "
            "All things arise from the Tao, flourish for a time, and then return. "
            "This return is called stillness, which is called returning to one's destiny. "
            "Rather than chasing constant change and novelty, the Taoist sage cultivates the ability "
            "to return to the quiet centre within — the place of clarity, simplicity, and deep knowing."
        ),
    },
    {
        "instruction": "What is the Taoist view of knowledge and wisdom?",
        "response": (
            "Taoism distinguishes between accumulated knowledge (learning) and genuine wisdom (understanding the Tao). "
            "'In pursuit of learning, every day something is added. In pursuit of the Tao, every day something is dropped.' "
            "True wisdom is not about accumulating facts but about shedding the ego, the grasping mind, "
            "and the compulsion to control. One who knows does not speak; one who speaks does not know. "
            "Knowing others is intelligence; knowing yourself is true wisdom. Mastering others is strength; mastering yourself is true power."
        ),
    },
    {
        "instruction": "How does Zhuangzi describe the ideal Taoist sage?",
        "response": (
            "In the Zhuangzi, the ideal sage moves through the world like wind through trees — "
            "present, responsive, and leaving no trace. The sage does not impose categories or fixed judgements "
            "on experience. They understand that all perspectives are relative: what is large from one view "
            "is small from another. The famous story of Cook Ding cutting the ox illustrates this — "
            "his knife never dulls because he follows the natural structure of the animal, not forcing his way. "
            "This is the Taoist mastery: moving effortlessly along the grain of reality."
        ),
    },
    {
        "instruction": "What does Taoism teach about the acceptance of death?",
        "response": (
            "Taoism views death not as a tragedy but as a natural transformation. "
            "When Zhuangzi's wife died, he was found singing. He explained: at first she had no life, "
            "then she came into being, took form, lived — and now she has returned. "
            "She sleeps peacefully in the great house of the universe. "
            "Clinging to life and fearing death is like a child refusing to go home at the end of the day. "
            "The sage accepts the endless transformations of existence without resistance or grief."
        ),
    },
    {
        "instruction": "What is the Taoist meaning of simplicity?",
        "response": (
            "Simplicity (樸, pu — the uncarved block) is one of Taoism's core values. "
            "The uncarved block represents potential in its purest form, before it has been shaped by "
            "ambition, social roles, or cultural conditioning. "
            "The Tao Te Ching advocates returning to this natural simplicity: fewer desires, less striving, "
            "plain living. This is not poverty of spirit but richness — the sage who desires nothing "
            "lacks nothing. 'Manifest plainness, embrace simplicity, reduce selfishness, have few desires.'"
        ),
    },
    {
        "instruction": "How does the Tao Te Ching describe the power of emptiness?",
        "response": (
            "The Tao Te Ching celebrates emptiness as the source of usefulness. "
            "Thirty spokes converge at the hub of a wheel, but it is the empty space at the centre "
            "that makes the wheel useful. Clay is shaped into a vessel, but it is the hollow space inside "
            "that makes it useful. Doors and windows are cut in walls, and it is the empty spaces "
            "that make the room useful. Therefore, what exists serves for profit, but what does not exist "
            "serves for utility. Emptiness is not absence — it is openness, receptivity, and potential."
        ),
    },
    {
        "instruction": "What does Taoism say about the dangers of desire and ambition?",
        "response": (
            "The Tao Te Ching warns that excessive desire and ambition are the root of suffering and conflict. "
            "'There is no greater misfortune than wanting more. There is no greater calamity than greed.' "
            "When we endlessly pursue wealth, status, and pleasure, we exhaust ourselves and harm others. "
            "The sage, by contrast, acts without grasping, leads without dominating, and achieves without taking credit. "
            "This is not resignation but a deep trust in the sufficiency of the present moment and the natural order."
        ),
    },
    {
        "instruction": "What is the significance of the three treasures in Taoism?",
        "response": (
            "The Tao Te Ching describes three treasures: compassion (慈, ci), frugality (儉, jian), "
            "and not daring to be first in the world (不敢為天下先). "
            "Compassion leads to courage — not the courage of aggression, but the courage to remain open and caring. "
            "Frugality creates abundance — by not squandering, one always has enough to give. "
            "Not presuming to lead allows one to become a true leader when the time comes. "
            "These three treasures are the practical expression of living in alignment with the Tao."
        ),
    },
    {
        "instruction": "How does Taoism approach conflict and competition?",
        "response": (
            "Taoism fundamentally opposes force, competition, and contention. "
            "The Tao Te Ching states: 'The way of heaven benefits without harming; the way of the sage acts without contending.' "
            "Like water, the Taoist does not fight to occupy high places but naturally finds the level. "
            "In conflict, the one who yields often wins. The soft overcomes the hard; the gentle overcomes the rigid. "
            "This is why the sage does not argue, does not compete for honour, and does not repay harm with harm — "
            "because contention itself is the loss."
        ),
    },
    {
        "instruction": "What can we learn from the Tao about living with uncertainty?",
        "response": (
            "The Tao teaches that uncertainty and change are the nature of reality, not aberrations to be fixed. "
            "'Fortune and misfortune take turns with each other' — what seems like disaster may become blessing, "
            "and vice versa. The sage therefore does not grasp tightly at any outcome or state. "
            "By remaining flexible, present, and empty of fixed expectations, "
            "one can respond appropriately to whatever arises. "
            "This is not passivity but a profound readiness — like an uncarved block, able to become anything."
        ),
    },
    {
        "instruction": "How does Taoist philosophy relate to nature?",
        "response": (
            "Nature is the primary teacher in Taoism. The Tao expresses itself through the patterns of mountains, rivers, "
            "seasons, and living things. Observing nature — how trees bend in wind without breaking, "
            "how rivers find their way around obstacles, how winter gives way to spring — "
            "reveals the principles by which the wise person should live. "
            "Human beings are not separate from nature but part of its flow. "
            "The Taoist sage acts as nature acts: purposefully but without forcing, consistently but without rigidity."
        ),
    },
    {
        "instruction": "What does the Tao Te Ching say about speaking and silence?",
        "response": (
            "The Tao Te Ching is paradoxically a book about the limits of language, beginning with the statement "
            "that the Tao which can be spoken is not the eternal Tao. "
            "Too many words exhaust the mind and often obscure rather than illuminate truth. "
            "'Those who know do not speak; those who speak do not know.' "
            "Silence is not emptiness of thought but fullness of presence. "
            "The sage communicates through being, through example, through the quality of their presence — "
            "not through argument, explanation, or self-promotion."
        ),
    },
    {
        "instruction": "What is the Taoist teaching on self-knowledge?",
        "response": (
            "Self-knowledge holds a central place in Taoist thought. "
            "'Knowing others is wisdom; knowing yourself is enlightenment. "
            "Mastering others requires force; mastering yourself requires strength.' "
            "The inward journey — understanding one's own nature, desires, habitual reactions, and attachments — "
            "is the foundation of Taoist practice. "
            "When one knows oneself deeply, action becomes effortless and appropriate, "
            "because it flows from genuine nature rather than from fear or ego."
        ),
    },
    {
        "instruction": "How can Taoist philosophy be applied to modern daily life?",
        "response": (
            "Taoist principles offer practical guidance for contemporary life. Wu wei suggests: "
            "before acting, pause and ask whether action is truly necessary, or whether non-interference "
            "would serve better. In work, allow projects to develop at their natural pace rather than forcing. "
            "In relationships, practice listening and yielding rather than always seeking to win. "
            "Simplify — reduce unnecessary possessions, commitments, and mental noise. "
            "Spend time in nature; let it recalibrate your sense of what is truly important. "
            "Notice the rhythms and patterns in your life, and work with them rather than against them."
        ),
    },
    {
        "instruction": "What is the Taoist perspective on the ego?",
        "response": (
            "Taoism views the over-assertive ego as the primary obstacle to living in harmony with the Tao. "
            "The ego is the part of us that insists on its separateness, its importance, its need to control. "
            "The sage does not lose their individuality but holds it lightly, like a cloud holds rain. "
            "By not insisting on being first, by not demanding recognition, by not clinging to fixed identity, "
            "the sage paradoxically accomplishes more and experiences greater freedom. "
            "'The Tao is like a well: used but never used up. Empty yet inexhaustible.' The ego that empties itself finds it is full."
        ),
    },
    {
        "instruction": "Explain the Taoist concept of yin and yang.",
        "response": (
            "Yin and yang are not opposing forces at war but complementary aspects of a single reality. "
            "Yin is associated with receptivity, darkness, softness, yielding, the moon, winter, and the feminine principle. "
            "Yang is associated with activity, light, hardness, assertion, the sun, summer, and the masculine principle. "
            "Each contains the seed of the other — this is why the yin-yang symbol shows a dot of light in the dark half "
            "and a dot of dark in the light half. All phenomena arise from the interplay of these two aspects. "
            "Health, harmony, and wisdom come from honouring both rather than insisting on one."
        ),
    },
    {
        "instruction": "What does the Tao Te Ching say about happiness and contentment?",
        "response": (
            "The Tao Te Ching locates contentment not in acquisition but in sufficiency. "
            "'He who knows that enough is enough will always have enough.' "
            "Chasing after wealth, fame, or pleasure is like trying to fill a leaking vessel — "
            "the more you add, the more escapes. True contentment arises when one aligns with the Tao "
            "and stops measuring life against an imagined future or an idealised standard. "
            "The sage enjoys what is present, wastes nothing in craving what is absent, "
            "and finds the extraordinary within the ordinary."
        ),
    },
    {
        "instruction": "What is the Taoist teaching on flexibility and resilience?",
        "response": (
            "Taoism prizes flexibility as the mark of life and rigidity as the mark of death. "
            "'A man is born gentle and supple. At his death he is hard and stiff. "
            "Green plants are tender and filled with sap. At their death they are withered and dry.' "
            "The tree that survives the storm is the one that bends. "
            "The sage cultivates this suppleness in mind and body: responding to each situation freshly, "
            "not locked into fixed positions or old habits. Resilience comes not from hardening against difficulty "
            "but from remaining fluid enough to flow around it."
        ),
    },
    {
        "instruction": "How does the Zhuangzi use stories and parables?",
        "response": (
            "The Zhuangzi is famous for its rich use of fables, parables, and apparently absurd scenarios "
            "to point beyond the limitations of ordinary thinking. Stories like the butterfly dream — "
            "'Am I a man dreaming I am a butterfly, or a butterfly dreaming I am a man?' — "
            "are designed not to be solved but to dissolve the rigid certainties of the conceptual mind. "
            "By making us laugh and wonder at the same time, Zhuangzi opens a space of radical openness "
            "where fixed categories soften and a more fluid, playful engagement with reality becomes possible."
        ),
    },
    {
        "instruction": "What does the Tao Te Ching say about war and violence?",
        "response": (
            "The Tao Te Ching is deeply opposed to violence and war. "
            "'Weapons are instruments of fear; they are not a wise man's tools.' "
            "The military leader who delights in conquest is unfit to lead, because victory through violence "
            "is always a tragedy, never a true triumph. "
            "Even in unavoidable conflict, the Taoist approach is to use minimum force, to not glorify killing, "
            "and to mourn those who fall on both sides. "
            "The greatest victories are those won without battle — through patience, non-contention, and moral authority."
        ),
    },
    {
        "instruction": "How does Taoism view the relationship between the individual and the universe?",
        "response": (
            "In Taoism, the individual is not separate from the universe but a particular expression of it. "
            "The same Tao that moves through the cosmos moves through the individual. "
            "This understanding dissolves the sense of alienation and separateness that causes so much suffering. "
            "When we act from our deepest nature — which is aligned with the Tao — our actions resonate "
            "with the whole. The Taoist does not seek to conquer the universe but to participate in it fully, "
            "humbly, and attentively, like a musician playing in an orchestra rather than trying to drown out all other instruments."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH SCRAPER  (ctext.org English translations)
# ─────────────────────────────────────────────────────────────────────────────

EN_SOURCES = [
    {"name": "Tao Te Ching (Legge)",   "url": "https://ctext.org/dao-de-jing/en"},
    {"name": "Zhuangzi (Legge)",        "url": "https://ctext.org/zhuangzi/en"},
    {"name": "Liezi (English)",         "url": "https://ctext.org/liezi/en"},
]

_MIN_EN_LEN = 40


def _clean_english(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_valid_english(text: str) -> bool:
    if len(text) < _MIN_EN_LEN:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.7:
        return False
    if re.search(r"\d{4,}", text):
        return False
    return True


def scrape_english_source(url: str, name: str, max_passages: int = 200) -> list[str]:
    headers = {"User-Agent": "TaoistLLM-Research/1.0 (Educational/Non-commercial)"}
    passages = []
    try:
        print(f"  Fetching {name} …", end=" ", flush=True)
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for elem in soup.select("td.en, div.english, p.english, span.en, td.text"):
            text = _clean_english(elem.get_text(separator=" ", strip=True))
            if _is_valid_english(text):
                passages.append(text)
                if len(passages) >= max_passages:
                    break
        print(f"got {len(passages)} passages.")
    except Exception as exc:
        print(f"FAILED ({exc})")
    time.sleep(1.0)
    return passages


def passages_to_pairs(passages: list[str]) -> list[dict]:
    """Wrap raw English passages into instruction-response pairs."""
    prompts = [
        "What does this Taoist passage mean?",
        "Explain this teaching from the Tao Te Ching.",
        "How should one interpret this Taoist wisdom?",
        "What practical guidance does this passage offer?",
        "Reflect on the deeper meaning of this Taoist teaching:",
    ]
    pairs = []
    seen = set()
    for p in passages:
        key = re.sub(r"\s+", "", p).lower()
        if key in seen:
            continue
        seen.add(key)
        prompt = prompts[hash(p) % len(prompts)]
        pairs.append({
            "instruction": f"{prompt}\n\n\"{p}\"",
            "response": (
                f"This passage reflects a core principle of Taoist thought. "
                f"It points to the way of the Tao — effortless, natural, and beyond conceptual grasping. "
                f"The teaching invites us to align our actions and intentions with the natural flow of reality, "
                f"releasing the need to force outcomes and trusting the process of life itself. "
                f"In practice, this means acting with presence, simplicity, and without unnecessary contention — "
                f"embodying the spirit of wu wei, the art of non-striving action."
            ),
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def pairs_to_jsonl_record(pair: dict) -> dict:
    """Format an instruction-response pair for the JSONL file."""
    return {
        "instruction": pair["instruction"],
        "response":    pair["response"],
        "text": f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['response']}",
    }


def main():
    parser = argparse.ArgumentParser(description="Build English Taoist instruction dataset locally.")
    parser.add_argument("--output",    default="taoist_data.jsonl", help="Output JSONL path")
    parser.add_argument("--preview",   action="store_true",         help="Print first 5 samples")
    parser.add_argument("--no-scrape", action="store_true",         help="Use seed corpus only (no network)")
    parser.add_argument("--max",       type=int, default=200,       help="Max passages per online source")
    args = parser.parse_args()

    print("=" * 65)
    print("  Taoist LLM — English Instruction Dataset Builder")
    print("=" * 65)

    pairs = list(SEED_PAIRS)

    if not args.no_scrape:
        print(f"\n[1/3] Scraping English translations ({len(EN_SOURCES)} sources) …")
        for src in EN_SOURCES:
            raw = scrape_english_source(src["url"], src["name"], max_passages=args.max)
            pairs.extend(passages_to_pairs(raw))
    else:
        print("\n[1/3] Skipping scrape (--no-scrape).")

    # Deduplicate by instruction text
    seen_instructions = set()
    deduped = []
    for p in pairs:
        key = re.sub(r"\s+", "", p["instruction"]).lower()
        if key not in seen_instructions:
            seen_instructions.add(key)
            deduped.append(p)

    print(f"\n[2/3] {len(deduped)} unique instruction-response pairs assembled.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in deduped:
            json.dump(pairs_to_jsonl_record(pair), f, ensure_ascii=False)
            f.write("\n")

    print(f"\n[3/3] Saved → {out_path.resolve()}  ({out_path.stat().st_size // 1024} KB)")
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
