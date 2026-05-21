#!/usr/bin/env python3
"""
Shared Taoist data-source classes.
Used by both scrape_data.py (local prep) and main.py (RunPod fallback).
"""

import os
import re
import time
import logging
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    "Reflect on this teaching in the spirit of wu wei (effortless action):",
    "How might water respond to the wisdom in this passage?",
    "What does this suggest about harmony with the Dao?",
    "Consider this from the perspective of natural spontaneity (ziran):",
    "In the tradition of Zhuangzi, how would you contemplate this idea?",
    "What practical wisdom does this Taoist passage offer?",
    "Explain this teaching from the Tao Te Ching.",
]

# ── Built-in fallback pairs (always present, no network needed) ───────────────

FALLBACK_PAIRS: List[Dict[str, str]] = [
    {"instruction": "What is the Tao?",
     "response": "The Tao (道) is the fundamental principle underlying all of reality. It cannot be fully captured in words — 'The Tao that can be told is not the eternal Tao.' It is the source from which all things arise, the nameless beginning of heaven and earth. Rather than a deity or a force to be commanded, it is the natural way of the universe itself: silent, effortless, and inexhaustible."},
    {"instruction": "What is wu wei?",
     "response": "Wu wei (無為) literally means 'non-doing' or 'effortless action.' It does not mean laziness or passivity, but acting in complete harmony with the natural flow of things — without force, without ego, without unnecessary interference. The sage accomplishes everything through wu wei: 'Do nothing, and nothing is left undone.'"},
    {"instruction": "What does the Tao Te Ching teach about water?",
     "response": "Water is one of the central metaphors in the Tao Te Ching. 'The highest good is like water' (上善若水). Water benefits all living things without competing, dwelling willingly in the lowest places that others disdain. Nothing in the world is softer than water, yet nothing surpasses it in overcoming the hard and strong."},
    {"instruction": "What does Taoism say about the relationship between opposites?",
     "response": "Taoism sees opposites not as contradictions but as complementary aspects of a single whole. Beauty only exists because we recognise ugliness; good only because we know bad. This interdependence is captured in the image of yin and yang: two forces that generate and balance each other in endless, dynamic harmony."},
    {"instruction": "What is the Taoist meaning of simplicity?",
     "response": "Simplicity (樸, pu — the uncarved block) is one of Taoism's core values. The uncarved block represents potential in its purest form, before it has been shaped by ambition, social roles, or cultural conditioning. 'Manifest plainness, embrace simplicity, reduce selfishness, have few desires.'"},
    {"instruction": "How should a leader govern according to Taoist philosophy?",
     "response": "The Tao Te Ching describes the ideal ruler as one who governs through stillness and restraint. 'Governing a large state is like cooking a small fish' — handle it too much and you ruin it. The best leader is barely noticed by the people; when the work is done the people say: 'We did it ourselves.'"},
    {"instruction": "What does Taoism teach about the acceptance of death?",
     "response": "Taoism views death not as a tragedy but as a natural transformation. When Zhuangzi's wife died, he was found singing. Clinging to life and fearing death is like a child refusing to go home at the end of the day. The sage accepts the endless transformations of existence without resistance."},
    {"instruction": "What does the Tao Te Ching say about the power of emptiness?",
     "response": "Thirty spokes converge at the hub of a wheel, but it is the empty space at the centre that makes the wheel useful. Clay is shaped into a vessel, but it is the hollow space inside that makes it useful. Therefore, what exists serves for profit, but what does not exist serves for utility."},
    {"instruction": "What is the Taoist view of knowledge and wisdom?",
     "response": "'In pursuit of learning, every day something is added. In pursuit of the Tao, every day something is dropped.' True wisdom is not about accumulating facts but about shedding the ego and the compulsion to control. Knowing others is intelligence; knowing yourself is true wisdom."},
    {"instruction": "How does Taoism approach conflict and competition?",
     "response": "The Tao Te Ching states: 'The way of heaven benefits without harming; the way of the sage acts without contending.' Like water, the Taoist does not fight for high places but naturally finds the level. The soft overcomes the hard; the gentle overcomes the rigid."},
    {"instruction": "What can we learn from the Tao about living with uncertainty?",
     "response": "The Tao teaches that uncertainty and change are the nature of reality. 'Fortune and misfortune take turns with each other.' The sage does not grasp tightly at any outcome. By remaining flexible, present, and empty of fixed expectations, one can respond appropriately to whatever arises."},
    {"instruction": "What is the significance of the three treasures in Taoism?",
     "response": "The Tao Te Ching describes three treasures: compassion (慈), frugality (儉), and not daring to be first in the world (不敢為天下先). Compassion leads to true courage; frugality creates abundance; not presuming to lead allows one to become a true leader when the time comes."},
    {"instruction": "How does Taoist philosophy relate to nature?",
     "response": "Nature is the primary teacher in Taoism. The Tao expresses itself through the patterns of mountains, rivers, seasons, and living things. Human beings are not separate from nature but part of its flow. The Taoist sage acts as nature acts: purposefully but without forcing, consistently but without rigidity."},
    {"instruction": "What is the Taoist teaching on self-knowledge?",
     "response": "'Knowing others is wisdom; knowing yourself is enlightenment. Mastering others requires force; mastering yourself requires strength.' The inward journey is the foundation of Taoist practice. When one knows oneself deeply, action becomes effortless and flows from genuine nature."},
    {"instruction": "How does the Zhuangzi describe the ideal Taoist sage?",
     "response": "In the Zhuangzi, the ideal sage moves through the world like wind through trees — present, responsive, and leaving no trace. The famous story of Cook Ding illustrates this: his knife never dulls because he follows the natural structure of the animal, not forcing his way. This is Taoist mastery: moving effortlessly along the grain of reality."},
    {"instruction": "How can Taoist philosophy be applied to modern daily life?",
     "response": "Wu wei suggests: before acting, pause and ask whether action is truly necessary. In work, allow projects to develop at their natural pace. In relationships, practice listening and yielding. Simplify — reduce unnecessary possessions and mental noise. Spend time in nature. Notice the rhythms in your life and work with them rather than against them."},
    {"instruction": "Explain the Taoist concept of yin and yang.",
     "response": "Yin and yang are complementary aspects of a single reality. Yin is associated with receptivity, darkness, softness, the moon, and winter. Yang with activity, light, hardness, the sun, and summer. Each contains the seed of the other. Health and wisdom come from honouring both rather than insisting on one."},
    {"instruction": "What does the Tao Te Ching say about happiness and contentment?",
     "response": "'He who knows that enough is enough will always have enough.' Chasing wealth, fame, or pleasure is like trying to fill a leaking vessel. True contentment arises when one aligns with the Tao and stops measuring life against an imagined ideal. The sage finds the extraordinary within the ordinary."},
    {"instruction": "What is the Taoist teaching on flexibility and resilience?",
     "response": "Taoism prizes flexibility as the mark of life and rigidity as the mark of death. 'A man is born gentle and supple. At his death he is hard and stiff.' The tree that survives the storm is the one that bends. Resilience comes not from hardening against difficulty but from remaining fluid enough to flow around it."},
    {"instruction": "What does the Tao Te Ching say about war and violence?",
     "response": "'Weapons are instruments of fear; they are not a wise man's tools.' The military leader who delights in conquest is unfit to lead. Even in unavoidable conflict, the Taoist approach is to use minimum force and mourn those who fall on both sides. The greatest victories are won without battle."},
]

# ── Base class ────────────────────────────────────────────────────────────────

class TaoistDataSource:
    """Base class for Taoist data collection."""

    def __init__(self, name: str, base_url: Optional[str] = None):
        self.name = name
        self.base_url = base_url
        self.collected: List[Dict[str, Any]] = []

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ── Source: Chinese Text Project ──────────────────────────────────────────────

class CTextSource(TaoistDataSource):
    """Classical Chinese texts from ctext.org (public domain)."""

    TEXTS = ["dao-de-jing", "zhuangzi", "liezi", "huainanzi"]

    def __init__(self):
        super().__init__("ctext", "https://ctext.org")

    def collect(self, texts: Optional[List[str]] = None, max_passages: int = 500) -> List[Dict[str, Any]]:
        texts = texts or self.TEXTS
        headers = {"User-Agent": "TaoistLLM-Research/2.0 (Educational)"}
        for text_id in texts:
            url = f"{self.base_url}/{text_id}"
            try:
                logger.info(f"  [ctext] Fetching {text_id} …")
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for p in soup.select("div.ctext, p.chinese, .passage"):
                    content = p.get_text(strip=True)
                    if re.match(r"^[\u4e00-\u9fff\s，。？！；：、""''（）《》\-]+$", content) and len(content) > 20:
                        prompt = PROMPT_TEMPLATES[hash(content) % len(PROMPT_TEMPLATES)]
                        self.collected.append({
                            "source": "ctext", "language": "classical_chinese",
                            "instruction": f"{prompt}\n\n{content}",
                            "response": (
                                "This classical passage embodies the Taoist way — effortless, natural, "
                                "and pointing beyond words to the living flow of the Tao. "
                                f"It teaches us: {content}"
                            ),
                        })
                        if len(self.collected) >= max_passages:
                            return self.collected
                time.sleep(0.8)
            except Exception as e:
                logger.warning(f"ctext scrape failed for {text_id}: {e}")
        return self.collected


# ── Source: Sacred Texts Archive ──────────────────────────────────────────────

class SacredTextsSource(TaoistDataSource):
    """Public-domain English translations from sacred-texts.com (Legge, Giles)."""

    TRANSLATIONS = {
        "tao/taote":  "Tao Te Ching (Legge)",
        "tao/chuang": "Zhuangzi (Legge)",
        "tao/lieh":   "Liezi (Giles)",
    }

    def __init__(self):
        super().__init__("sacred_texts", "https://sacred-texts.com")

    def collect(self, max_per_text: int = 80) -> List[Dict[str, Any]]:
        for path, title in self.TRANSLATIONS.items():
            url = f"{self.base_url}/{path}.htm"
            try:
                logger.info(f"  [sacred_texts] Fetching {title} …")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                body = soup.find("div", class_="main") or soup.body
                if not body:
                    continue
                raw = body.get_text(separator="\n", strip=True)
                raw = re.sub(r"\[.*?\]|\(c\).*", "", raw)
                paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 60]
                count = 0
                for para in paragraphs:
                    if count >= max_per_text:
                        break
                    prompt = PROMPT_TEMPLATES[hash(para) % len(PROMPT_TEMPLATES)]
                    self.collected.append({
                        "source": "sacred_texts", "language": "english",
                        "instruction": f"{prompt}\n\n\"{para}\"",
                        "response": (
                            f"This passage from {title} reflects a core Taoist principle. "
                            "It invites us to align with the natural flow of the Tao — acting without forcing, "
                            "knowing without grasping, and finding strength in yielding. "
                            "In practice, this means embodying wu wei: effortless action arising from "
                            "one's deepest, most natural state of being."
                        ),
                    })
                    count += 1
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"sacred_texts fetch failed for {path}: {e}")
        return self.collected


# ── Source: Nanhuaijin HuggingFace dataset ────────────────────────────────────

class NanhuaijinHFSource(TaoistDataSource):
    """Nanhuaijin scholarly commentaries from HuggingFace (requires HF_TOKEN)."""

    def __init__(self):
        super().__init__("nanhuaijin_hf")

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.info("  [nanhuaijin] HF_TOKEN not set — skipping.")
            return []
        try:
            from datasets import load_dataset as _load_ds
            logger.info("  [nanhuaijin] Loading from HuggingFace …")
            ds = _load_ds("wobure/nanhuaijin-collections", split="train", token=token)
            for item in ds:
                content = item.get("content", "")[:1400]
                if len(content) < 30:
                    continue
                prompt = PROMPT_TEMPLATES[hash(content) % len(PROMPT_TEMPLATES)]
                self.collected.append({
                    "source": "nanhuaijin_hf", "language": "modern_chinese",
                    "instruction": f"{prompt}\n\n{content}",
                    "response": f"In the spirit of Master Nan Huaijin's commentaries: {content[:300]}…",
                })
            logger.info(f"  [nanhuaijin] Loaded {len(self.collected)} passages.")
        except Exception as e:
            logger.warning(f"nanhuaijin load failed: {e}")
        return self.collected


# ── Source: Documentary metadata ──────────────────────────────────────────────

class DocumentaryMetadataSource(TaoistDataSource):
    """Documentary summaries from Films For Action (metadata only)."""

    TITLES = ["The Art of Effortless Living", "The Art of Letting Go", "Opening Dao"]

    def __init__(self):
        super().__init__("documentaries", "https://www.filmsforaction.org")

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        for title in self.TITLES:
            url = f"{self.base_url}/search/?q={title.replace(' ', '+')}"
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                meta = soup.find("meta", attrs={"name": "description"})
                desc = soup.find("p", class_="description")
                summary = (
                    (meta.get("content") if meta else None)
                    or (desc.get_text(strip=True) if desc else None)
                )
                if summary and len(summary) > 40:
                    self.collected.append({
                        "source": "documentaries", "language": "english",
                        "instruction": f"What is the Taoist teaching explored in the documentary '{title}'?",
                        "response": f"The documentary '{title}' explores: {summary}",
                    })
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Documentary metadata failed for '{title}': {e}")
        return self.collected


# ── Registry ──────────────────────────────────────────────────────────────────

SOURCE_MAP: Dict[str, type] = {
    "ctext":         CTextSource,
    "sacred_texts":  SacredTextsSource,
    "nanhuaijin":    NanhuaijinHFSource,
    "documentaries": DocumentaryMetadataSource,
}
