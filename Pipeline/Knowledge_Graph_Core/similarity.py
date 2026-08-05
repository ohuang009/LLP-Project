from __future__ import annotations

from collections import Counter
import math
import re


WORD_RE = re.compile(r"[a-z0-9]+")


def features(text: str) -> Counter[str]:
    lowered = text.casefold()
    words = WORD_RE.findall(lowered)
    values: Counter[str] = Counter(f"w:{word}" for word in words)
    compact = " ".join(words)
    for n in (3, 4, 5):
        for index in range(max(0, len(compact) - n + 1)):
            values[f"c:{compact[index:index+n]}"] += 0.35
    return values


def idf_for(texts: list[str]) -> dict[str, float]:
    documents = [set(features(text)) for text in texts]
    count = max(len(documents), 1)
    df: Counter[str] = Counter()
    for document in documents:
        df.update(document)
    return {term: math.log((1 + count) / (1 + frequency)) + 1 for term, frequency in df.items()}


def vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = features(text)
    total = sum(counts.values()) or 1.0
    return {term: (frequency / total) * idf.get(term, 1.0) for term, frequency in counts.items()}


def cosine_vectors(left: dict[str, float], right: dict[str, float]) -> float:
    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine(left: str, right: str) -> float:
    idf = idf_for([left, right])
    return cosine_vectors(vector(left, idf), vector(right, idf))


def matrix_vectors(texts: list[str]) -> list[dict[str, float]]:
    idf = idf_for(texts)
    return [vector(text, idf) for text in texts]
