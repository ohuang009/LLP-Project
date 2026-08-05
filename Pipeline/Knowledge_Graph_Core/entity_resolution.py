from __future__ import annotations

from collections import defaultdict
import re

import numpy as np

from .embeddings import SentenceEmbeddingBackend
from .models import stable_id


SCOPED_DISTINCT_LABELS = {"Observation", "Claim", "Condition"}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _acronym(value: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", value)
              if token.casefold() not in {"of", "and", "the", "for", "to"}]
    return "".join(token[0] for token in tokens).casefold() if len(tokens) >= 2 else ""


def _acronym_match(left: str, right: str) -> bool:
    lnorm = re.sub(r"[^a-z0-9]", "", left.casefold())
    rnorm = re.sub(r"[^a-z0-9]", "", right.casefold())
    return bool(lnorm and rnorm and (lnorm == _acronym(right) or rnorm == _acronym(left)))


def _embedding_text(mention: dict) -> str:
    return f"Name: {mention['surface_text']}. Definition: {mention['definition']}"


class EntityResolver:
    def __init__(self, config: dict, embedding_backend=None):
        self.config = config["entity_resolution"]
        self.embedding_backend = embedding_backend or SentenceEmbeddingBackend(
            model_name=self.config["embedding_model"],
            batch_size=int(self.config.get("embedding_batch_size", 64)),
        )

    def resolve(self, mentions: list[dict]) -> tuple[list[dict], dict]:
        comparable = [m for m in mentions if m["label"] != "Publication"]
        if not comparable:
            return [], self.calibrate([])

        embeddings = np.asarray(
            self.embedding_backend.encode([_embedding_text(m) for m in comparable]),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(comparable):
            raise ValueError("Embedding backend returned an invalid matrix shape")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        embedding_by_id = {
            mention["mention_id"]: embeddings[index]
            for index, mention in enumerate(comparable)
        }

        by_label: dict[str, list[dict]] = defaultdict(list)
        for mention in comparable:
            by_label[mention["label"]].append(mention)

        decisions: list[dict] = []
        max_candidates = int(self.config["maximum_candidates_per_node"])
        for label, label_mentions in sorted(by_label.items()):
            label_matrix = np.stack([
                embedding_by_id[mention["mention_id"]] for mention in label_mentions
            ])
            for index, mention in enumerate(label_mentions):
                if index == 0:
                    continue
                current = label_matrix[index]
                prior_mentions = label_mentions[:index]
                prior_matrix = label_matrix[:index]
                similarities = prior_matrix @ current
                candidate_count = min(max_candidates, len(prior_mentions))
                if candidate_count < len(prior_mentions):
                    candidate_indexes = np.argpartition(similarities, -candidate_count)[-candidate_count:]
                else:
                    candidate_indexes = np.arange(len(prior_mentions))
                candidate_indexes = sorted(
                    candidate_indexes,
                    key=lambda prior_index: (
                        -float(similarities[prior_index]),
                        prior_mentions[prior_index]["mention_id"],
                    ),
                )
                for prior_index in candidate_indexes:
                    prior = prior_mentions[int(prior_index)]
                    embedding_score = float(np.clip(similarities[int(prior_index)], -1.0, 1.0))
                    acronym_same = _acronym_match(mention["surface_text"], prior["surface_text"])
                    exact_name = mention["normalized_name"] == prior["normalized_name"]
                    decisions.append(self._decision(
                        prior, mention, label, embedding_score,
                        acronym_same, exact_name,
                    ))

        unique = {decision["resolution_id"]: decision for decision in decisions}
        values = sorted(unique.values(), key=lambda item: item["resolution_id"])
        return values, self.calibrate(values)

    def _decision(
        self,
        left: dict,
        right: dict,
        label: str,
        embedding_score: float,
        acronym_same: bool,
        exact_name: bool,
    ) -> dict:
        high = float(self.config["high_review_threshold"])
        uncertain = float(self.config["uncertain_review_threshold"])
        confidence_score = max(0.0, min(1.0, embedding_score))
        if confidence_score >= high:
            confidence_level = "high"
        elif confidence_score >= uncertain:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        if label in SCOPED_DISTINCT_LABELS:
            band = "low"
            action = "keep_distinct"
            reason = "ontology_policy_scoped_result_or_condition"
        elif confidence_score >= high:
            band = "high"
            action = "review_high_confidence_match"
            reason = "embedding_cosine_above_high_review_threshold"
        elif confidence_score >= uncertain:
            band = "uncertain_middle"
            action = "review_uncertain_match"
            reason = "embedding_cosine_inside_uncertain_review_band"
        else:
            band = "low"
            action = "keep_distinct"
            reason = "embedding_cosine_below_review_threshold"
        return {
            "resolution_id": stable_id("resolution", left["mention_id"], right["mention_id"]),
            "label": label,
            "left_mention_id": left["mention_id"],
            "right_mention_id": right["mention_id"],
            "left_surface_text": left["surface_text"],
            "right_surface_text": right["surface_text"],
            "embedding_model": self.embedding_backend.model_name,
            "embedding_cosine": round(embedding_score, 6),
            "cosine_distance": round(1.0 - embedding_score, 6),
            "confidence_score": round(confidence_score, 6),
            "confidence_level": confidence_level,
            "score_basis": "embedding_cosine",
            # Compatibility alias for existing queue sorting and downstream bundles.
            "combined_cosine": round(confidence_score, 6),
            "exact_name_match": exact_name,
            "acronym_match": acronym_same,
            "band": band,
            "action": action,
            "reason": reason,
            "auto_merged": False,
            "thresholds": {"high": high, "uncertain": uncertain},
        }

    @staticmethod
    def calibrate(decisions: list[dict]) -> dict:
        silver = []
        for decision in decisions:
            left = decision["left_surface_text"].casefold().strip()
            right = decision["right_surface_text"].casefold().strip()
            same = decision["exact_name_match"] or decision["acronym_match"]
            distinct = not same and not (_tokens(left) & _tokens(right))
            if same or distinct:
                silver.append((decision["confidence_score"], same))
        candidates = [round(value / 100, 2) for value in range(35, 96, 2)]
        results = []
        for threshold in candidates:
            tp = fp = tn = fn = 0
            for score, same in silver:
                predicted = score >= threshold
                if predicted and same:
                    tp += 1
                elif predicted and not same:
                    fp += 1
                elif not predicted and not same:
                    tn += 1
                else:
                    fn += 1
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            balanced = ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2
            results.append({
                "threshold": threshold,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "balanced_accuracy": round(balanced, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })
        best = max(results, key=lambda item: (
            item["balanced_accuracy"], item["f1"], item["threshold"]
        )) if results else None
        return {
            "method": "embedding_cosine_with_silver_exact_acronym_and_token_disjoint_labels",
            "warning": "Confidence is a similarity-based review score, not a calibrated probability of identity.",
            "silver_pair_count": len(silver),
            "recommended_review_threshold": best["threshold"] if best else None,
            "best_result": best,
            "threshold_sweep": results,
        }
