from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .ontology import Ontology
from .similarity import cosine


REQUIRED_ARTIFACTS = {
    "manifest.json",
    "mentions.jsonl",
    "node_review_candidates.jsonl",
    "llm_node_judge_calls.jsonl",
    "llm_discovered_nodes.jsonl",
    "new_ontology_class_candidates.jsonl",
    "assertions.jsonl",
    "relationship_candidates.jsonl",
    "relationship_unknown_nodes.jsonl",
    "llm_relationship_judge_calls.jsonl",
    "resolution_decisions.jsonl",
    "rejections.jsonl",
    "review_high_confidence.jsonl",
    "review_uncertain_middle.jsonl",
    "keep_distinct_low.jsonl",
}


def validate_run(
    output_dir: Path,
    ontology: Ontology,
    config: dict,
    documents: list[dict],
    mentions: list[dict],
    assertions: list[dict],
    resolutions: list[dict],
) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    for name in sorted(REQUIRED_ARTIFACTS):
        if not (output_dir / name).is_file():
            errors.append({"code": "missing_artifact", "artifact": name})

    sentence_index: dict[str, dict] = {}
    paragraph_ids: set[str] = set()
    section_ids: set[str] = set()
    per_document_sentences: Counter[str] = Counter()
    for item in documents:
        document_id = item["pipeline_document_id"]
        parsed = item["parsed"]
        seen: set[str] = set()
        for section in parsed.get("sections", []):
            if section["id"] in seen:
                errors.append({"code": "duplicate_source_id", "id": section["id"]})
            seen.add(section["id"])
            section_ids.add(section["id"])
            for paragraph in section.get("paragraphs", []):
                if paragraph["id"] in seen:
                    errors.append({"code": "duplicate_source_id", "id": paragraph["id"]})
                seen.add(paragraph["id"])
                paragraph_ids.add(paragraph["id"])
                for sentence in paragraph.get("sentences", []):
                    if sentence["id"] in seen or sentence["id"] in sentence_index:
                        errors.append({"code": "duplicate_source_id", "id": sentence["id"]})
                    seen.add(sentence["id"])
                    sentence_index[sentence["id"]] = sentence
                    per_document_sentences[document_id] += 1
        minimum = int(config["validation"]["minimum_sentences_per_document"])
        if per_document_sentences[document_id] < minimum:
            errors.append({
                "code": "too_few_sentences",
                "document_id": document_id,
                "actual": per_document_sentences[document_id],
                "minimum": minimum,
            })

    mention_index: dict[str, dict] = {}
    per_document_labels: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        mention_id = mention["mention_id"]
        if mention_id in mention_index:
            errors.append({"code": "duplicate_mention_id", "id": mention_id})
        mention_index[mention_id] = mention
        label = mention["label"]
        if label not in ontology.nodes:
            errors.append({"code": "unknown_node_label", "mention_id": mention_id, "label": label})
        per_document_labels[mention["document_id"]].add(label)
        source = mention["source"]
        if source["kind"] == "sentence_span":
            sentence = sentence_index.get(source["sentence_id"])
            if sentence is None:
                errors.append({"code": "missing_sentence_reference", "mention_id": mention_id})
                continue
            start, end = source["start_char"], source["end_char"]
            if not (0 <= start < end <= len(sentence["text"])):
                errors.append({"code": "invalid_character_span", "mention_id": mention_id})
            elif sentence["text"][start:end] != source["quote"] or source["quote"] != mention["surface_text"]:
                errors.append({"code": "evidence_quote_mismatch", "mention_id": mention_id})
            if source["paragraph_id"] not in paragraph_ids or source["section_id"] not in section_ids:
                errors.append({"code": "missing_parent_reference", "mention_id": mention_id})

    assertion_ids: set[str] = set()
    per_document_edges: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    for assertion in assertions:
        assertion_id = assertion["assertion_id"]
        if assertion_id in assertion_ids:
            errors.append({"code": "duplicate_assertion_id", "id": assertion_id})
        assertion_ids.add(assertion_id)
        source = mention_index.get(assertion["subject_mention_id"])
        target = mention_index.get(assertion["object_mention_id"])
        if source is None or target is None:
            errors.append({"code": "missing_assertion_endpoint", "assertion_id": assertion_id})
            continue
        valid = ontology.valid_relationship(assertion["predicate"], source["label"], target["label"])
        if not valid or not assertion.get("ontology_valid"):
            errors.append({"code": "domain_range_violation", "assertion_id": assertion_id})
        for sentence_id in assertion["evidence_sentence_ids"]:
            if sentence_id not in sentence_index:
                errors.append({"code": "missing_assertion_evidence", "assertion_id": assertion_id, "sentence_id": sentence_id})
        per_document_edges[assertion["document_id"]] += 1
        predicate_counts[assertion["predicate"]] += 1

    resolution_ids: set[str] = set()
    resolution_bands: Counter[str] = Counter()
    for decision in resolutions:
        if decision["resolution_id"] in resolution_ids:
            errors.append({"code": "duplicate_resolution_id", "id": decision["resolution_id"]})
        resolution_ids.add(decision["resolution_id"])
        if decision["left_mention_id"] not in mention_index or decision["right_mention_id"] not in mention_index:
            errors.append({"code": "missing_resolution_endpoint", "resolution_id": decision["resolution_id"]})
        if decision.get("auto_merged"):
            errors.append({"code": "unexpected_automatic_merge", "resolution_id": decision["resolution_id"]})
        resolution_bands[decision["band"]] += 1

    expectation_results = []
    expectations = ontology.data.get("corpus_expectations", [])
    minimum_recall = float(config["validation"]["minimum_document_node_recall"])
    for item in documents:
        document_id = item["pipeline_document_id"]
        title = item["parsed"]["document"].get("title") or item["parsed"]["document"]["filename"]
        ranked = sorted(
            ((cosine(title, expected["title"]), expected) for expected in expectations),
            key=lambda pair: pair[0],
            reverse=True,
        )
        title_score, expected = ranked[0]
        expected_labels = set(expected["expected_node_labels"])
        actual_labels = per_document_labels[document_id]
        matched = expected_labels & actual_labels
        recall = len(matched) / max(len(expected_labels), 1)
        result = {
            "document_id": document_id,
            "parsed_title": title,
            "matched_expectation": expected["short_name"],
            "title_cosine": round(title_score, 4),
            "expected_labels": sorted(expected_labels),
            "actual_labels": sorted(actual_labels),
            "matched_labels": sorted(matched),
            "missing_labels": sorted(expected_labels - actual_labels),
            "document_label_recall": round(recall, 4),
        }
        expectation_results.append(result)
        if title_score < 0.35:
            warnings.append({"code": "weak_expectation_title_match", **result})
        if recall < minimum_recall:
            errors.append({"code": "node_coverage_below_threshold", **result, "minimum": minimum_recall})
        if config["validation"].get("require_edge_per_document") and per_document_edges[document_id] == 0:
            errors.append({"code": "no_relationships_extracted", "document_id": document_id})

    return {
        "schema_version": "1.0",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "documents": len(documents),
            "sentences": len(sentence_index),
            "mentions": len(mentions),
            "assertions": len(assertions),
            "resolution_decisions": len(resolutions),
            "node_label_counts": dict(sorted(Counter(m["label"] for m in mentions).items())),
            "predicate_counts": dict(sorted(predicate_counts.items())),
            "resolution_bands": dict(sorted(resolution_bands.items())),
            "average_document_node_recall": round(
                sum(item["document_label_recall"] for item in expectation_results) / max(len(expectation_results), 1), 4
            ),
        },
        "document_expectation_checks": expectation_results,
    }
