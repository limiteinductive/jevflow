"""
Issue #3: Sentence splitting, paragraph grouping, and AI marker detection.
"""

import re
from dataclasses import dataclass, field


# Known AI marker phrases — detected by regex, not Jev (converge fix F5)
AI_MARKERS = [
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "furthermore",
    "moreover",
    "in conclusion",
    "in summary",
    "delve",
    "delving",
    "multifaceted",
    "it's worth mentioning",
    "plays a crucial role",
    "serves as a testament",
    "a testament to",
    "landscape of",
    "paradigm shift",
    "holistic approach",
    "leverage",
    "leveraging",
    "streamline",
    "streamlined",
    "foster",
    "fostering",
    "navigate the complexities",
    "at the forefront",
    "in today's rapidly evolving",
    "it cannot be overstated",
    "the importance of .+ cannot be overstated",
]

_MARKER_PATTERN = re.compile(
    "|".join(re.escape(m) if "+" not in m else m for m in AI_MARKERS),
    re.IGNORECASE,
)


@dataclass
class Sentence:
    text: str
    start: int  # char offset into original
    end: int    # char offset into original
    index: int  # sentence index
    paragraph: int  # paragraph index
    markers: list[str] = field(default_factory=list)


@dataclass
class Paragraph:
    text: str
    start: int
    end: int
    index: int
    sentences: list[Sentence] = field(default_factory=list)


def split_sentences(text: str) -> list[Sentence]:
    """
    Split text into sentences with character offsets; a sentence ends at ". ", "! ", "? " or a line break.
    Uses regex that handles common abbreviations.
    """
    paragraphs = split_paragraphs(text)
    sentences: list[Sentence] = []
    sent_idx = 0

    for para in paragraphs:
        para_text = para.text
        raw_parts = re.split(r'(?<=[.!?])\s+|\s*\n\s*', para_text)
        # Rejoin false splits on abbreviations
        parts = []
        abbrevs = {'dr', 'mr', 'mrs', 'ms', 'prof', 'jr', 'sr', 'st', 'vs', 'etc', 'e.g', 'i.e', 'u.s', 'u.k'}
        for part in raw_parts:
            if parts:
                words = parts[-1].rstrip('.').split()
                last_word = words[-1].lower() if words else ''
                if last_word in abbrevs:
                    parts[-1] = parts[-1] + ' ' + part
                    continue
            parts.append(part)

        offset = para.start
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Find actual position in original text
            pos = text.find(part, offset)
            if pos == -1:
                pos = offset

            # Detect AI markers
            markers = _MARKER_PATTERN.findall(part)

            sent = Sentence(
                text=part if part.endswith(('.', '!', '?')) else part + '.',
                start=pos,
                end=pos + len(part),
                index=sent_idx,
                paragraph=para.index,
                markers=[m.lower() for m in markers],
            )
            sentences.append(sent)
            para.sentences.append(sent)
            sent_idx += 1
            offset = pos + len(part)

    return sentences


def split_paragraphs(text: str) -> list[Paragraph]:
    """Split text into paragraphs on double newlines, preserving offsets."""
    paragraphs: list[Paragraph] = []
    # Split on one or more blank lines
    parts = re.split(r'\n\s*\n', text)
    offset = 0
    for idx, part in enumerate(parts):
        part_stripped = part.strip()
        if not part_stripped:
            continue
        pos = text.find(part_stripped, offset)
        if pos == -1:
            pos = offset
        para = Paragraph(
            text=part_stripped,
            start=pos,
            end=pos + len(part_stripped),
            index=idx,
        )
        paragraphs.append(para)
        offset = pos + len(part_stripped)
    return paragraphs


def get_paragraph_context(sentence: Sentence, paragraphs: list[Paragraph]) -> str:
    """Get the paragraph containing this sentence, for context."""
    for para in paragraphs:
        if para.index == sentence.paragraph:
            return para.text
    return sentence.text


def count_words(text: str) -> int:
    return len(text.split())


def detect_markers(text: str) -> list[str]:
    """Detect AI marker phrases in text."""
    return [m.lower() for m in _MARKER_PATTERN.findall(text)]
