from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from .llm_node_judge import MandatoryLLMNodeJudge
from .models import normalize_name, stable_id
from .ontology import Ontology


RESULT_RE = re.compile(
    r"\b(?:achiev(?:e|ed|es)|reduc(?:e|ed|tion)|improv(?:e|ed|ement)|increas(?:e|ed)|"
    r"decreas(?:e|ed)|outperform(?:s|ed)?|yield(?:s|ed)?|result(?:s|ed)?|measur(?:e|ed))\b",
    re.I,
)
NUMBER_RE = re.compile(
    r"(?P<qualifier>up to|greater than|less than|approximately|about|nearly|over|under)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|weeks?|days?|years?|hours?|"
    r"mg/L|Âµg/L|kg|g|mg|ppm|ppb|Â°C|K|kWh|MPa|Pa)?",
    re.I,
)
CLAIM_RE = re.compile(
    r"\b(?:we (?:show|demonstrate|conclude|recommend|find)|results? (?:show|indicate|demonstrate)|"
    r"this (?:study|work) (?:shows|demonstrates|establishes|suggests)|suggests? that|"
    r"conclud(?:e|ed)|recommend(?:s|ed)?|demonstrat(?:e|ed|es))\b",
    re.I,
)
CHALLENGE_RE = re.compile(
    r"\b(?:constrained by|limited by|barrier(?:s)? (?:to|for)|challenge(?:s)? (?:of|in|for)|"
    r"limitation(?:s)? (?:of|include|in))\s+(?P<value>[^,.;:]{3,100})",
    re.I,
)
GENERIC_PATTERNS = {
    "AISystem": [
        re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-|\s)?(?:Agentic|Agent|RAG|MAS|LLaMA)\b"),
        re.compile(r"\b(?:multi-agent|agent-based|agentic) (?:framework|mechanism|system|platform)\b", re.I),
    ],
    "Model": [
        re.compile(r"\b(?:large language|machine[- ]learning|graph neural|predictive|simulation|statistical|hydraulic) models?\b", re.I),
        re.compile(r"\bGPT[- ]?\d(?:\.\d+)?\b", re.I),
    ],
    "Agent": [re.compile(r"\b(?:[A-Za-z][A-Za-z-]*\s+){0,3}agents?\b", re.I)],
    "Dataset": [re.compile(r"\b(?:[A-Za-z][A-Za-z0-9-]*\s+){0,4}(?:dataset|benchmark|corpus)\b", re.I)],
    "Task": [
        re.compile(
            r"\b(?:[A-Za-z][A-Za-z-]*\s+){0,4}(?:scheduling|optimization|prediction|retrieval|"
            r"extraction|simulation|classification|prioritization|forecasting|monitoring|calibration)\b",
            re.I,
        )
    ],
    "Process": [re.compile(r"\b(?:[A-Za-z][A-Za-z-]*\s+){0,3}(?:treatment|production|degradation|desalination|oxidation)\b", re.I)],
    "Chemical": [re.compile(r"\b(?:CO2|CH4|N2O|PFAS|ROS)\b", re.I)],
    "PolicyDocument": [re.compile(r"\bISO\s?\d{4,5}\b", re.I)],
}
LEADING_SPAN_NOISE_RE = re.compile(
    r"^(?:(?:and|or|the|a|an|to|by|with|from|between|of|in|on|for|each|"
    r"these|those|their|our|its)\s+)+",
    re.I,
)


@dataclass(frozen=True)
class SpanCandidate:
    label: str
    start: int
    end: int
    method: str
    base_confidence: float


class NodeExtractor:
    def __init__(
        self,
        ontology: Ontology,
        lexicon_path: Path,
        config: dict,
        llm_judge: MandatoryLLMNodeJudge,
    ) -> None:
        self.ontology = ontology
        self.lexicons: dict[str, list[str]] = json.loads(lexicon_path.read_text(encoding="utf-8"))
        self.config = config
        self.llm_judge = llm_judge
        self._compiled_lexicons = {
            label: [
                re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.I)
                for phrase in sorted(phrases, key=len, reverse=True)
            ]
            for label, phrases in self.lexicons.items()
        }

    def extract(self, parsed: dict, pipeline_document_id: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        audit_start = len(self.llm_judge.audit_records)
        mentions: list[dict] = []
        review_candidates: list[dict] = []
        rejections: list[dict] = []
        document = parsed["document"]
        title = document.get("title") or document["filename"]
        mentions.append(self._metadata_mention(
            pipeline_document_id, "Publication", title, "document.title", 1.0,
            {"filename": document["filename"]},
        ))
        for position, author in enumerate(self._split_authors(document.get("authors_text"))):
            mentions.append(self._metadata_mention(
                pipeline_document_id, "Person", author, "document.authors_text", 0.98,
                {"author_position": position + 1},
            ))

        sentence_records: list[dict] = []
        candidate_index: dict[str, dict] = {}
        sentence_context: dict[str, dict] = {}
        for section in parsed.get("sections", []):
            for paragraph in section.get("paragraphs", []):
                for sentence in paragraph.get("sentences", []):
                    candidates, candidate_rejections = self._sentence_candidates(
                        pipeline_document_id, sentence, paragraph, section
                    )
                    rejections.extend(candidate_rejections)
                    for candidate in candidates:
                        candidate_index[candidate["candidate_id"]] = candidate
                    sentence_context[sentence["id"]] = {
                        "sentence": sentence, "paragraph": paragraph, "section": section,
                    }
                    sentence_records.append({
                        "sentence_id": sentence["id"],
                        "section_title": section["title"],
                        "text": sentence["text"],
                        "candidates": [self._candidate_for_prompt(row) for row in candidates],
                    })

        judgments = self.llm_judge.judge(sentence_records)
        accepted_keys: set[tuple[str, int, int, str]] = set()
        for sentence_id, result in judgments.items():
            context = sentence_context[sentence_id]
            audit = result.get("_llm_audit", {})
            for judgment in result["candidate_judgments"]:
                candidate = candidate_index[judgment["candidate_id"]]
                decision_record = self._decision_record(judgment, audit, candidate)
                if judgment["decision"] == "accept":
                    mention = self._mention_from_candidate(candidate, judgment, decision_record)
                    key = self._mention_key(mention)
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        mentions.append(mention)
                elif judgment["decision"] == "review":
                    review_candidates.append({**candidate, "candidate_validation": decision_record})
                else:
                    rejections.append({
                        "rejection_id": stable_id("reject", candidate["candidate_id"], audit.get("request_hash", "")),
                        "stage": "llm_candidate_judgment",
                        "reason": judgment["reason"],
                        **candidate,
                        "candidate_validation": decision_record,
                    })

            for discovery in result["discoveries"]:
                mention = self._mention_from_discovery(
                    pipeline_document_id, context, discovery, audit
                )
                key = self._mention_key(mention)
                if key not in accepted_keys:
                    accepted_keys.add(key)
                    mentions.append(mention)

            for proposal in result["new_class_candidates"]:
                review_candidates.append(self._new_class_review(
                    pipeline_document_id, context, proposal, audit
                ))

        for audit in self.llm_judge.audit_records[audit_start:]:
            for invalid in audit.get("invalid_outputs", []):
                raw_output = invalid.get("raw_output", {})
                rejections.append({
                    "rejection_id": stable_id(
                        "reject", audit.get("request_hash", ""),
                        invalid.get("sentence_id", ""), invalid.get("output_type", ""),
                        raw_output.get("surface_text", ""),
                    ),
                    "stage": "llm_output_validation",
                    "reason": invalid.get("reason", "invalid LLM output"),
                    "document_id": pipeline_document_id,
                    "sentence_id": invalid.get("sentence_id", ""),
                    "surface_text": raw_output.get("surface_text", ""),
                    "raw_output": raw_output,
                    "candidate_validation": {
                        "outcome": "reject",
                        "judge": "exact_source_evidence_validator",
                        "provider": audit.get("provider", ""),
                        "model": audit.get("model", ""),
                        "response_id": audit.get("response_id", ""),
                        "request_hash": audit.get("request_hash", ""),
                        "prompt_version": audit.get("prompt_version", ""),
                        "llm_used": True,
                    },
                })

        return mentions, review_candidates, rejections, list(self.llm_judge.audit_records[audit_start:])

    @staticmethod
    def _split_authors(authors: str | None) -> list[str]:
        if not authors:
            return []
        authors = re.sub(r"\s+and\s+", ",", authors, flags=re.I)
        return [part.strip(" ,;*") for part in authors.split(",") if len(part.strip(" ,;*")) > 2]

    def _metadata_mention(self, document_id: str, label: str, text: str, field: str,
                          confidence: float, attributes: dict) -> dict:
        mention_id = stable_id("mention", document_id, field, label, text)
        attributes = {**attributes, "candidate_validation": {
            "outcome": "accept", "judge": "structural_metadata_rule", "llm_used": False,
            "reason": "Ontology-mandated publication metadata node.",
        }}
        return {
            "mention_id": mention_id, "document_id": document_id, "label": label,
            "surface_text": text, "normalized_name": normalize_name(text),
            "definition": self._definition(text, label, f"Document metadata field {field}."),
            "confidence": confidence, "status": "accepted", "extraction_method": "metadata_rule",
            "ontology_matches": [{"label": label, "cosine": 1.0}], "attributes": attributes,
            "source": {"kind": "document_metadata", "field": field},
        }

    def _sentence_candidates(self, document_id: str, sentence: dict, paragraph: dict,
                             section: dict) -> tuple[list[dict], list[dict]]:
        text = sentence["text"]
        candidates: list[SpanCandidate] = []
        for label, patterns in self._compiled_lexicons.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    candidates.append(SpanCandidate(label, match.start(), match.end(), "specific_lexicon", 0.94))
        for label, patterns in GENERIC_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    candidates.append(SpanCandidate(label, match.start(), match.end(), "rule_candidate", 0.72))
        challenge = CHALLENGE_RE.search(text)
        if challenge:
            candidates.append(SpanCandidate("Challenge", challenge.start("value"), challenge.end("value"), "challenge_candidate", 0.72))
        numbers = list(NUMBER_RE.finditer(text))
        for match in numbers:
            if match.group("unit") or match.group("qualifier"):
                candidates.append(SpanCandidate("Condition", match.start(), match.end(), "quantity_candidate", 0.72))
        if numbers and RESULT_RE.search(text):
            candidates.append(SpanCandidate("Observation", 0, len(text), "result_candidate", 0.72))
        if CLAIM_RE.search(text):
            candidates.append(SpanCandidate("Claim", 0, len(text), "claim_candidate", 0.72))

        grouped: dict[tuple[int, int], list[tuple[SpanCandidate, list[dict]]]] = {}
        for candidate in candidates:
            raw_surface = text[candidate.start:candidate.end]
            leading_trim = 0
            if candidate.method == "rule_candidate":
                leading = LEADING_SPAN_NOISE_RE.match(raw_surface)
                leading_trim = leading.end() if leading else 0
            surface = raw_surface[leading_trim:].strip(" ,;:()")
            if len(surface) < 2:
                continue
            start = text.find(surface, candidate.start + leading_trim, candidate.end + 1)
            adjusted = SpanCandidate(candidate.label, start, start + len(surface), candidate.method, candidate.base_confidence)
            match_text = f"{surface}. Context: {text}. Section: {section['title']}."
            scores = self.ontology.class_scores(match_text)
            grouped.setdefault((adjusted.start, adjusted.end), []).append((adjusted, scores))

        selected: list[dict] = []
        rejections: list[dict] = []
        for (start, end), alternatives in grouped.items():
            alternatives.sort(key=lambda item: (item[0].base_confidence, item[0].label), reverse=True)
            chosen, scores = alternatives[0]
            surface = text[start:end]
            candidate_id = stable_id("candidate", document_id, sentence["id"], start, end, chosen.label, chosen.method)
            selected.append({
                "candidate_id": candidate_id, "document_id": document_id,
                "sentence_id": sentence["id"], "paragraph_id": paragraph["id"],
                "section_id": section["id"], "section_title": section["title"],
                "suggested_label": chosen.label, "surface_text": surface,
                "evidence_sentence": text,
                "normalized_name": normalize_name(surface), "candidate_method": chosen.method,
                "base_confidence": chosen.base_confidence, "ontology_matches": scores[:5],
                "source": {
                    "kind": "sentence_span", "sentence_id": sentence["id"],
                    "paragraph_id": paragraph["id"], "section_id": section["id"],
                    "section_title": section["title"], "pages": sentence.get("pages", []),
                    "start_char": start, "end_char": end, "quote": surface,
                    "provenance_precision": "exact_sentence_characters; page_set_inherited_from_paragraph",
                },
            })
            for rejected, rejected_scores in alternatives[1:]:
                rejections.append({
                    "rejection_id": stable_id("reject", candidate_id, rejected.label),
                    "stage": "rule_candidate_generation", "reason": "ambiguous_span_lower_ranked_label",
                    "candidate_label": rejected.label, "chosen_label": chosen.label,
                    "surface_text": surface, "sentence_id": sentence["id"],
                    "ontology_matches": rejected_scores[:5],
                })
        return selected, rejections

    @staticmethod
    def _candidate_for_prompt(candidate: dict) -> dict:
        return {
            "candidate_id": candidate["candidate_id"],
            "surface_text": candidate["surface_text"],
            "start_char": candidate["source"]["start_char"],
            "end_char": candidate["source"]["end_char"],
            "suggested_label": candidate["suggested_label"],
            "candidate_method": candidate["candidate_method"],
        }

    @staticmethod
    def _decision_record(judgment: dict, audit: dict, candidate: dict) -> dict:
        return {
            "outcome": judgment["decision"], "reason": judgment["reason"],
            "confidence": judgment["confidence"], "canonical_name": judgment["canonical_name"],
            "ontology_label": judgment["ontology_label"], "initial_label": candidate["suggested_label"],
            "judge": f"{audit.get('provider', 'unknown')}_structured_llm",
            "provider": audit.get("provider", ""),
            "model": audit.get("model", ""), "response_id": audit.get("response_id", ""),
            "request_hash": audit.get("request_hash", ""), "prompt_version": audit.get("prompt_version", ""),
            "prompt_sha256": audit.get("prompt_sha256", ""), "llm_used": True,
        }

    def _mention_from_candidate(self, candidate: dict, judgment: dict, decision: dict) -> dict:
        label = judgment["ontology_label"]
        source = candidate["source"]
        mention_id = stable_id("mention", candidate["document_id"], source["sentence_id"],
                               source["start_char"], source["end_char"], label)
        attributes = self._attributes(label, candidate["surface_text"], candidate["surface_text"])
        attributes.update({
            "candidate_validation": decision, "canonical_name": judgment["canonical_name"],
            "candidate_id": candidate["candidate_id"], "candidate_method": candidate["candidate_method"],
            "lexicon_status": "specific_anchor" if candidate["candidate_method"] == "specific_lexicon" else "rule_candidate",
        })
        ontology_matches = list(candidate["ontology_matches"])
        if not any(row["label"] == label for row in ontology_matches):
            selected = next(
                (row for row in self.ontology.class_scores(
                    f"{candidate['surface_text']}. {judgment['definition']}"
                ) if row["label"] == label),
                {"label": label, "cosine": 0.0},
            )
            ontology_matches.append(selected)
        ontology_matches.sort(key=lambda row: float(row["cosine"]), reverse=True)
        return {
            "mention_id": mention_id, "document_id": candidate["document_id"], "label": label,
            "surface_text": candidate["surface_text"], "normalized_name": normalize_name(judgment["canonical_name"] or candidate["surface_text"]),
            "definition": (
                f"{judgment['canonical_name'] or candidate['surface_text']} ({label}). "
                f"LLM definition: {judgment['definition']} "
                f"Evidence context: {candidate['evidence_sentence']}"
            ),
            "status": "accepted", "extraction_method": "llm_judged_candidate",
            "ontology_matches": ontology_matches, "attributes": attributes, "source": source,
        }

    def _mention_from_discovery(self, document_id: str, context: dict, discovery: dict, audit: dict) -> dict:
        sentence, paragraph, section = context["sentence"], context["paragraph"], context["section"]
        label = discovery["ontology_label"]
        start, end = int(discovery["start_char"]), int(discovery["end_char"])
        surface = discovery["surface_text"]
        mention_id = stable_id("mention", document_id, sentence["id"], start, end, label)
        decision = {
            "outcome": "accept", "reason": discovery["reason"], "confidence": discovery["confidence"],
            "canonical_name": discovery["canonical_name"], "ontology_label": label,
            "judge": f"{audit.get('provider', 'unknown')}_structured_llm",
            "provider": audit.get("provider", ""), "model": audit.get("model", ""),
            "response_id": audit.get("response_id", ""), "request_hash": audit.get("request_hash", ""),
            "prompt_version": audit.get("prompt_version", ""), "prompt_sha256": audit.get("prompt_sha256", ""),
            "llm_used": True, "discovered_outside_lexicon": True,
        }
        source = {
            "kind": "sentence_span", "sentence_id": sentence["id"], "paragraph_id": paragraph["id"],
            "section_id": section["id"], "section_title": section["title"],
            "pages": sentence.get("pages", []), "start_char": start, "end_char": end,
            "quote": surface, "provenance_precision": "exact_sentence_characters; page_set_inherited_from_paragraph",
        }
        return {
            "mention_id": mention_id, "document_id": document_id, "label": label,
            "surface_text": surface, "normalized_name": normalize_name(discovery["canonical_name"] or surface),
            "definition": (
                f"{discovery['canonical_name'] or surface} ({label}). "
                f"LLM definition: {discovery['definition']} "
                f"Evidence context: {sentence['text']}"
            ),
            "status": "accepted", "extraction_method": "llm_discovery",
            "ontology_matches": self.ontology.class_scores(f"{surface}. {sentence['text']}")[:5],
            "attributes": {"candidate_validation": decision, "canonical_name": discovery["canonical_name"],
                           "lexicon_status": "discovered_outside_specific_lexicon"},
            "source": source,
        }

    @staticmethod
    def _new_class_review(document_id: str, context: dict, proposal: dict, audit: dict) -> dict:
        sentence, paragraph, section = context["sentence"], context["paragraph"], context["section"]
        return {
            "candidate_id": stable_id("new-class", document_id, sentence["id"], proposal["start_char"],
                                      proposal["end_char"], proposal["proposed_class_name"]),
            "document_id": document_id, "sentence_id": sentence["id"], "paragraph_id": paragraph["id"],
            "section_id": section["id"], "section_title": section["title"],
            "surface_text": proposal["surface_text"], "normalized_name": normalize_name(proposal["surface_text"]),
            "proposed_class_name": proposal["proposed_class_name"], "definition": proposal["definition"],
            "source": {
                "kind": "sentence_span", "sentence_id": sentence["id"], "paragraph_id": paragraph["id"],
                "section_id": section["id"], "section_title": section["title"],
                "pages": sentence.get("pages", []), "start_char": proposal["start_char"],
                "end_char": proposal["end_char"], "quote": proposal["surface_text"],
            },
            "candidate_validation": {
                "outcome": "review", "reason": proposal["reason"], "confidence": proposal["confidence"],
                "judge": f"{audit.get('provider', 'unknown')}_structured_llm",
                "provider": audit.get("provider", ""), "model": audit.get("model", ""),
                "response_id": audit.get("response_id", ""), "request_hash": audit.get("request_hash", ""),
                "prompt_version": audit.get("prompt_version", ""), "prompt_sha256": audit.get("prompt_sha256", ""),
                "llm_used": True, "decision_type": "new_ontology_class_candidate",
            },
        }

    @staticmethod
    def _mention_key(mention: dict) -> tuple[str, int, int, str]:
        source = mention["source"]
        return (source.get("sentence_id", ""), int(source.get("start_char", -1)),
                int(source.get("end_char", -1)), mention["label"])

    def _attributes(self, label: str, surface: str, sentence: str) -> dict:
        if label not in {"Observation", "Condition"}:
            return {}
        match = NUMBER_RE.search(surface if label == "Condition" else sentence)
        return {} if not match else {
            "original_value": match.group("value"), "original_unit": match.group("unit"),
            "qualifier": match.group("qualifier"),
        }

    def _definition(self, surface: str, label: str, context: str) -> str:
        return f"{surface} is extracted as {label}: {self.ontology.nodes[label]['definition']} Source context: {context}"
