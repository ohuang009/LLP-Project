from __future__ import annotations

from collections import defaultdict

from .models import normalize_name


def build_review_queues(decisions: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for decision in decisions:
        names = sorted([
            normalize_name(decision["left_surface_text"]),
            normalize_name(decision["right_surface_text"]),
        ])
        key = (decision["band"], decision["label"], names[0], names[1])
        grouped[key].append(decision)

    queues = {"high": [], "uncertain_middle": [], "low": []}
    for (band, label, left_name, right_name), values in grouped.items():
        values.sort(key=lambda item: (-item["combined_cosine"], item["resolution_id"]))
        best = values[0]
        queues[band].append({
            "queue_id": f"queue_{best['resolution_id'].split('_', 1)[-1]}",
            "band": band,
            "action": best["action"],
            "label": label,
            "left_name": left_name,
            "right_name": right_name,
            "best_combined_cosine": best["combined_cosine"],
            "minimum_combined_cosine": min(item["combined_cosine"] for item in values),
            "maximum_combined_cosine": max(item["combined_cosine"] for item in values),
            "decision_count": len(values),
            "sample_resolution_ids": [item["resolution_id"] for item in values[:5]],
            "auto_merged": False,
        })
    for values in queues.values():
        values.sort(key=lambda item: (-item["best_combined_cosine"], item["label"], item["left_name"], item["right_name"]))
    return queues
