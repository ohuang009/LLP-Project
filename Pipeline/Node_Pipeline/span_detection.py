from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
from typing import Callable, Iterable

from .common import GENERIC_TERMS, normalized, stable_id

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    method: str
    confidence: float
    canonical_id: str = ""
    canonical_name: str = ""
    metadata: dict | None = None

RULES: dict[str, tuple[str, ...]] = {
    "AISystem": (
        r"(?<![A-Za-z0-9'’])(?-i:[A-Z])[A-Za-z0-9]*(?:[- ](?-i:[A-Z])[A-Za-z0-9]*){0,4}(?:RAG|MAS|(?<!M)Agentic|Copilot)\b",
    ),
    "Model": (
        r"\b(?:openai|google|qwen|meta-llama|anthropic)/(?=[A-Za-z0-9._-]*(?:\s*-\s*)?\d)[A-Za-z0-9._]+(?:\s*-\s*[A-Za-z0-9._]+)*\b",
        r"\b(?:GPT[- ]?\d(?:\.\d+)?(?:o|\s+Search)?|Llama[- ]?\d(?:\.\d+)?(?:\s*(?:-|âˆ’)+\s*\d+\s*B)?|DeepSeek(?:[- ]?(?:V?\d+(?:\.\d+)?|R1))?|Qwen(?:\s*-\s*VL(?:\s*-\s*Max)?|[- ]?VL(?:[- ]?Max)?|[- ]?\d+(?:\.\d+)?)?|BERT|RoBERTa|XGBoost|Random Forest)\b",
        r"\b(?:BAAI/bge-[A-Za-z0-9._-]+|Voyage reranker-[A-Za-z0-9._-]+|Jina reranker v\d+|Qwen\d* reranker)\b",
    ),
    "Tool": (
        r"(?<![A-Za-z0-9-])(?:EPANET|WNTR|SCADA|GIS|SWMM|WaterGEMS|BioWin|GPS-X|MATLAB|Python|RStudio)(?![A-Za-z0-9-])",
    ),
    "Dataset": (
        r"\b(?-i:[A-Z])[A-Za-z0-9_-]{2,}(?:\s+(?-i:[A-Z])[A-Za-z0-9_-]{2,}){0,3}\s+(?:dataset|benchmark|database)\b",
        r"\b(?:domain-specific|selective|curated|tailored)?\s*(?:knowledge base|database)\s+of\s+\d+[^.;]{0,90}?(?:studies|articles|references)\b",
        r"\b\d+\s+(?:question(?:\s*[âˆ’-]\s*answer)?|Q&A)\s+(?:data\s*sets?|datasets?)\b",
        r"\b(?:L-Town|C-Town|Net1|Net2|Net3)\b",
        r"\b(?:test\s+split\s+(?:comprising|containing)\s+)?(?:\d{1,3}(?:,\s*\d{3})+|\d+)\s+(?:Monte\s+Carlo\s+)?(?:training\s+|test\s+)?(?:sample\s+data|samples?)\b",
        r"\b(?:benchmark\s+dataset\s+(?:of|with)\s+)?(?:eleven|\d+)\s+(?:distinct\s+)?WDSs?\b",
    ),
    "TreatmentProcess": (
        r"\b(?:granular\s+activated\s+carbon\s+)?(?:adsorption|ozonation|chlorination|coagulation|flocculation|filtration|disinfection|desalination|anaerobic\s+digestion|activated\s+sludge(?:\s+process)?|reverse\s+osmosis|membrane\s+bioreactor|advanced\s+oxidation(?:\s+process)?)\b",
    ),
    "TreatmentUnit": (
        r"\b(?:membrane\s+bioreactor|anaerobic\s+digester|aeration\s+tank|clarifier|sedimentation\s+tank|filter\s+bed|reverse\s+osmosis\s+unit)\b",
    ),
    "WaterMatrix": (
        r"\b(?:raw\s+water|drinking\s+water|potable\s+water|reclaimed\s+water|recycled\s+water|wastewater\s+influent|influent\s+wastewater|treated\s+effluent|secondary\s+effluent|tertiary\s+effluent|sewage\s+sludge|brine|concentrate)\b",
    ),
    "WaterSystem": (
        r"\b(?:water\s+distribution\s+(?:systems?|networks?)|WDSs?|WDNs?|groundwater\s+systems?|aquifer\s+systems?)\b",
    ),
    "WaterSource": (
        r"\b(?:groundwater|aquifers?|surface\s+water|source\s+water|rivers?|lakes?)\b",
    ),
    "TreatmentPlant": (
        r"\b(?:(?:drinking\s+water|wastewater)\s+treatment\s+plants?|DWTPs?|WWTPs?)\b",
    ),
    "Contaminant": (
        r"\b(?:PFAS|PFOS|PFOA|nitrate|nitrite|ammonia|phosphate|microplastics?|arsenic|lead|cadmium|mercury|chromium)\b",
    ),
    "Metric": (
        r"\b(?:removal\s+efficiency|effluent\s+concentration|energy\s+(?:use|consumption|intensity)|greenhouse\s+gas\s+emissions?|root\s+mean\s+square\s+error|mean\s+absolute\s+error|RMSE|MAE)\b",
        r"\b(?:answer\s+(?:correctness|relevance)(?:\s+pass)?\s+rate|correctness\s+pass\s+rate|context\s+recall|faithfulness|factual(?:ity|\s+correctness)|retrieval\s+quality|citation(?:-supported)?\s+(?:quality|consistency)|inference\s+latenc(?:y|ies)|operational\s+costs?|token\s+usage)\b",
        r"\b(?:task\s+success\s+rate|success\s+rate|tool\s+invocation\s+accuracy|numerical\s+(?:accuracy|correctness)|engineering-grade\s+reliability|human\s+intervention\s+rate)\b",
        r"\b(?:(?:overall|prediction|classification|tool\s+invocation)\s+)?accuracy\b",
        r"\b(?:normalized\s+mean\s+absolute\s+error|simulation\s+speedup|eutrophication\s+potential|operational\s+cost|effluent\s+quality)\b",
    ),
    "Parameter": (
        r"\b(?:pH|temperature|hydraulic\s+retention\s+time|solids\s+retention\s+time|HRT|SRT|flow\s+rate|membrane\s+flux|contact\s+time|operating\s+pressure|dissolved\s+oxygen|chemical\s+dos(?:e|age))\b",
    ),
    "Standard": (
        r"\b(?:ISO\s*\d{3,}(?:[-:]\d+)*|ASTM\s+[A-Z]\d+(?:-\d+)?|AWWA\s+[A-Z]\d+(?:-\d+)?|EPA\s+(?:Method|Rule|Standard)\s+[A-Z0-9-]+)\b",
    ),
    "Agent": (
        r"\b(?:Orchestrator|TaskExecutor|CodeRunner|DataAnalyzer|Knowledge\s+Agent|Modelling\s+Agent|Modeling\s+Agent)\b",
        r"\b(?:retrieval|review|evaluation|orchestrator|task\s+executor|code\s+runner|data\s+analyzer)\s+(?:sub-)?agent\b",
        r"\b(?:Orchestrating|Hydraulic\s+Simulation|Coding|Optimization|Planning|Knowledge|Modelling|Modeling)\s+Agent\b",
        r"\b[A-Z][A-Za-z0-9]*(?:Coder|Planner|Analyzer|Executor|Orchestrator)\b",
    ),
    "Task": (
        r"\b(?:wastewater\s+expert-level\s+question\s+answering|question\s+answering|literature\s+review(?:\s+generation)?|review\s+generation|engineering\s+(?:decision|scientific)\s+support|carbon\s+emissions\s+management|wastewater\s+treatment\s+(?:questions|tasks|problems))\b",
        r"\b(?:natural\s+language-controlled\s+WDN\s+simulation\s+and\s+analysis|WDNs?\s+control|hydraulic\s+(?:and\s+water\s+quality\s+)?simulation|water\s+quality\s+simulation|result\s+analysis|workflow\s+planning|simulation\s+control|system\s+characteristics?|system\s+dynamics|system\s+operation|scenario\s+simulation|model\s+calibration|pump\s+scheduling|code\s+generation)\b",
        r"\b(?:hydraulic\s+state\s+estimation|real-time\s+optimal\s+control)\b",
    ),
    "Method": (
        r"\b(?:semantic\s+segmentation|semantic\s+chunking|reranking\s+mechanism|optimized\s+retrieval|iterative\s+multiagent\s+framework|retrieval-augmented\s+generation|RAG|multi-agent\s+(?:orchestration|pipeline)|human-in-the-loop(?:\s+(?:mechanism|approach|oversight|mode))?|tool-driven\s+nested\s+design)\b",
        r"\b(?:multi-agent\s+deep\s+reinforcement\s+learning|life\s+cycle\s+assessment|improved\s+(?:graph\s+neural\s+network|GNN)\s+architecture|adapted\s+physics-informed\s+algorithm|iterative\s+two-phase\s+training\s+scheme|physics-preserving\s+(?:data\s+)?normalization)\b",
    ),
    "Challenge": (
        r"\b(?:information\s+silos?|lack\s+of\s+professional\s+domain\s+knowledge|insufficient\s+capability\s+to\s+solve\s+complex\s+engineering\s+problems|hallucination(?:s|\s+risks?)?|weak\s+retrieval\s+quality|non-?tailored\s+knowledge\s+base|fabricated\s+citation\s+links?|net-zero\s+challenge|lack(?:s|ing)?\s+(?:the\s+)?quantitative,?\s+predictive,?\s+and\s+evaluative\s+capabilities|missing\s+knowledge\s+coverage|retrieval\s+failure|reasoning\s+errors?|operational\s+complexity|dependence\s+on\s+specialized\s+expertise|predefined(?:\.inp|\s+input)\s+files|offline\s+operation)\b",
        r"\b(?:exploding\s+gradients?|unsupported\s+pumps?\s+and\s+pressure-reducing\s+valves?|static-in-time\s+hydraulic\s+state\s+estimation|large\s+(?:number\s+of\s+samples|training-data\s+requirement)|long\s+and\s+computationally\s+intensive\s+training)\b",
    ),
}

# The former suffix list included the generic acronym "MAS". Case-insensitive
# matching then classified names such as "Thomas" as AI systems. Keep only
# distinctive product-name endings; other system names are discovered from
# explicit introduce/propose constructions below.
RULES["AISystem"] = (
    r"(?<![A-Za-z0-9'])(?-i:[A-Z])[A-Za-z0-9]*(?:[- ](?-i:[A-Z])[A-Za-z0-9]*){0,4}(?:(?-i:RAG)|(?-i:(?<!M)Agentic)|(?-i:Copilot))\b",
)


INTRODUCED_SYSTEM_PATTERNS = (
    re.compile(
        r"\b(?:(?:This|The)\s+(?:study|paper|work)\s+|We\s+)?"
        r"(?:introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|present(?:s|ed|ing)?|develop(?:s|ed|ing)?)\s+"
        r"(?:an?\s+|the\s+)?(?P<name>[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\s*,?\s+"
        r"(?:an?\s+|the\s+)?[^.;]{0,65}\b(?:framework|architecture|system|platform|pipeline)\b"
    ),
    re.compile(
        r"\b(?P<name>[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\s*,?\s+"
        r"(?:is|was)\s+(?:introduced|proposed|presented|developed|described)\s+as\s+"
        r"(?:an?\s+|the\s+)?[^.;]{0,55}\b(?:framework|architecture|system|platform|pipeline)\b"
    ),
)

def _specific_system_name(value: str) -> bool:
    compact = value.strip(" ,.;:()")
    return bool(
        len(compact) >= 4
        and normalized(compact) not in GENERIC_TERMS
        and (
            "-" in compact
            or any(char.isupper() for char in compact[1:])
            or (compact.isupper() and any(char.isalpha() for char in compact))
        )
    )


def introduced_system_spans(text: str) -> list[Span]:
    """Discover a named AI system from an explicit introduce/propose construction."""
    spans: list[Span] = []
    for pattern in INTRODUCED_SYSTEM_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group("name")
            suffix = text[match.end("name"):match.end("name") + 12]
            if re.match(r"\s+et\s+al\b", suffix, re.I):
                continue
            if not _specific_system_name(name):
                continue
            spans.append(Span(
                match.start("name"), match.end("name"), "AISystem", "grammatical_pattern", .96,
                canonical_id=stable_id("entity", "AISystem", normalized(name)),
                canonical_name=name,
            ))
    return spans


def document_system_alias_spans(text: str, names: Iterable[str]) -> list[Span]:
    spans: list[Span] = []
    for name in names:
        expression = re.escape(name).replace(r"\-", r"\s*-\s*")
        for match in re.finditer(rf"(?<![A-Za-z0-9]){expression}(?![A-Za-z0-9])", text, re.I):
            spans.append(Span(
                match.start(), match.end(), "AISystem", "document_alias", .94,
                canonical_id=stable_id("entity", "AISystem", normalized(name)),
                canonical_name=name,
            ))
    return spans


def document_model_alias_spans(text: str, names: Iterable[str]) -> list[Span]:
    """Propagate only model names first grounded by ontology construction."""
    spans: list[Span] = []
    for name in names:
        expression = re.escape(name).replace(r"\-", r"\s*-\s*")
        for match in re.finditer(rf"(?<![A-Za-z0-9]){expression}(?![A-Za-z0-9])", text, re.I):
            spans.append(Span(
                match.start(), match.end(), "Model", "document_alias", .94,
                canonical_id=stable_id("entity", "Model", normalized(name)),
                canonical_name=name,
            ))
    return spans


def rule_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for label, patterns in RULES.items():
        for source in patterns:
            for match in re.finditer(source, text, flags=re.I):
                surface = match.group(0).strip()
                if label == "AISystem" and re.match(r"^(?:and|or|the|a|an|s)\s+", surface, re.I):
                    continue
                if label == "Dataset" and normalized(surface) in {"the benchmark", "a benchmark", "the dataset", "a dataset"}:
                    continue
                if label == "Model" and normalized(surface) == "deepseek" and re.match(r"[- ]AI\b", text[match.end():], re.I):
                    continue
                if label == "Contaminant" and normalized(surface) == "lead":
                    nearby = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    if not re.search(r"\b(?:Pb|heavy\s+metal|contaminant|concentration|mg\s*/\s*L|[µu]g\s*/\s*L)\b", nearby, re.I):
                        continue
                if normalized(surface) not in GENERIC_TERMS:
                    canonical_name = surface
                    canonical_id = ""
                    normalized_surface = normalized(surface)
                    if label == "Agent":
                        agent_names = {
                            "orchestrator": "Orchestrator",
                            "taskexecutor": "TaskExecutor",
                            "task executor": "TaskExecutor",
                            "coderunner": "CodeRunner",
                            "code runner": "CodeRunner",
                            "dataanalyzer": "DataAnalyzer",
                            "data analyzer": "DataAnalyzer",
                            "orchestrating": "Orchestrating Agent",
                            "hydraulic simulation": "Hydraulic Simulation Agent",
                            "coding": "Coding Agent",
                            "optimization": "Optimization Agent",
                            "planning": "Planning Agent",
                        }
                        without_role = re.sub(r"\s+(?:sub-)?agent$", "", surface, flags=re.I)
                        canonical_name = agent_names.get(normalized(without_role), surface)
                        canonical_id = stable_id("entity", "Agent", normalized(canonical_name))
                    elif label == "Tool":
                        tool_names = {
                            "epanet": "EPANET", "wntr": "WNTR", "scada": "SCADA",
                            "gis": "GIS", "python": "Python", "matlab": "MATLAB",
                        }
                        canonical_name = tool_names.get(normalized_surface, surface)
                        canonical_id = stable_id("entity", "Tool", normalized(canonical_name))
                    elif label == "Model" and "/" in surface:
                        canonical_name = re.sub(r"\s*-\s*", "-", surface)
                        canonical_id = stable_id("entity", "Model", normalized(canonical_name))
                    elif label == "Model" and normalized_surface.startswith("qwen"):
                        compact = re.sub(r"\s*-\s*", "-", surface)
                        canonical_name = compact.replace(" ", "-") if normalized_surface in {"qwen vl", "qwen vl max"} else compact
                        canonical_id = stable_id("entity", "Model", normalized(canonical_name))
                    elif label == "Method" and normalized_surface.startswith("human in the loop"):
                        canonical_name = "human-in-the-loop"
                        canonical_id = stable_id("entity", "Method", normalized(canonical_name))
                    elif label == "Method" and normalized_surface in {"rag", "retrieval augmented generation"}:
                        canonical_name = "retrieval-augmented generation"
                        canonical_id = stable_id("entity", "Method", normalized(canonical_name))
                    elif label == "Task" and normalized_surface in {"system characteristic", "system characteristics"}:
                        canonical_name = "System Characteristics"
                        canonical_id = stable_id("entity", "Task", normalized(canonical_name))
                    elif label == "WaterSystem" and normalized_surface in {
                        "wds", "wdss", "water distribution system", "water distribution systems",
                    }:
                        canonical_name = "water distribution system"
                        canonical_id = stable_id("entity", "WaterSystem", normalized(canonical_name))
                    elif label == "Challenge":
                        challenge_names = {
                            "hallucinations": "hallucination",
                            "hallucination risk": "hallucination",
                            "hallucination risks": "hallucination",
                            "reasoning errors": "reasoning error",
                            "predefined inp files": "predefined input files",
                            "predefined input files": "predefined input files",
                        }
                        canonical_name = challenge_names.get(normalized_surface, surface)
                        canonical_id = stable_id("entity", "Challenge", normalized(canonical_name))
                    elif label == "Metric" and "correctness" in normalized_surface and "rate" in normalized_surface:
                        canonical_name = "answer correctness rate"
                        canonical_id = stable_id("entity", "Metric", normalized(canonical_name))
                    elif label == "Metric" and normalized_surface in {
                        "accuracy", "overall accuracy", "prediction accuracy", "classification accuracy",
                    }:
                        canonical_name = "accuracy"
                        canonical_id = stable_id("entity", "Metric", normalized(canonical_name))
                    elif label == "Dataset" and "7637" in normalized_surface:
                        owner = re.search(r"\b([A-Z][A-Za-z0-9-]*RAG)\b", text)
                        canonical_name = f"{owner.group(1)} knowledge base" if owner else "domain knowledge base"
                        canonical_id = stable_id("entity", "Dataset", normalized(canonical_name))
                    spans.append(Span(
                        match.start(), match.end(), label, "environmental_profile_pattern", .78,
                        canonical_id=canonical_id, canonical_name=canonical_name,
                    ))
    spans.extend(introduced_system_spans(text))
    return spans


def select_spans(spans: list[Span], text: str) -> list[Span]:
    unique: dict[tuple[int, int, str], Span] = {}
    priority = {
        "persistent_lexicon": 6,
        "persistent_lexicon_ambiguous": 5,
        "scibert_svo_argument": 4,
    }
    for span in spans:
        surface = text[span.start:span.end].strip(" ,;:()")
        if len(surface) < 2:
            continue
        key = (span.start, span.end, span.label)
        current = unique.get(key)
        if current is None or priority.get(span.method, 0) > priority.get(current.method, 0):
            unique[key] = span
    # Keep overlapping spans when their boundaries differ. Candidate generation is
    # deliberately recall-oriented; DeepSeek or a human resolves nested options.
    return sorted(unique.values(), key=lambda row: (row.start, row.end, row.label))
