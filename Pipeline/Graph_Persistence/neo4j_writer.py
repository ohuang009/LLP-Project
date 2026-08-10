from __future__ import annotations

from collections import defaultdict
import json
import re
from urllib import error, request

from Pipeline.core import ONTOLOGY_PATH, SCOPED_LABELS, json_read, jsonl_read, now_iso, stable_id


NEO4J_COMMIT_URL = "http://127.0.0.1:7477/db/neo4j/tx/commit"
SAFE_RELATIONSHIP = re.compile(r"^[A-Z][A-Z0-9_]*$")
SAFE_LABEL = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
EMBEDDED_PROVENANCE_CLASSES = {"Mention", "EvidenceFragment"}


def _post(statements: list[dict], *, timeout: int = 45) -> dict:
    """Handle post for this stage. It helps persist validated nodes, evidence, and relationships in Neo4j."""
    payload = json.dumps({"statements": statements}, ensure_ascii=False).encode("utf-8")
    call = request.Request(
        NEO4J_COMMIT_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(call, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "The Engineered Water Systems Neo4j instance is unavailable on localhost:7477. "
            "Start it, then run the paper again."
        ) from exc
    if result.get("errors"):
        details = "; ".join(item.get("message", "Neo4j write failed") for item in result["errors"])
        raise RuntimeError(details)
    return result


def _statement(source: str, rows: list[dict] | None = None) -> dict:
    """Handle statement for this stage. It helps persist validated nodes, evidence, and relationships in Neo4j."""
    return {"statement": source, "parameters": {"rows": rows or []}}


def apply_ontology_labels(entity_rows: list[dict]) -> dict:
    """Apply exactly one concrete ontology label to every extracted entity."""
    ontology = json_read(ONTOLOGY_PATH)
    allowed_labels = set(ontology["nodes"])
    entity_ids = [row["id"] for row in entity_rows]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in entity_rows:
        label = row["label"]
        if label not in allowed_labels or not SAFE_LABEL.fullmatch(label):
            raise ValueError(f"Unknown or unsafe ontology class for entity {row['id']}: {label}")
        grouped[label].append({"id": row["id"]})

    # Remove stale ontology labels only from entities in this run, then apply the
    # authoritative label. This makes reclassification idempotent and visible.
    label_statements: list[dict] = []
    for label in sorted(allowed_labels):
        if not SAFE_LABEL.fullmatch(label):
            raise ValueError(f"Unsafe ontology class label: {label}")
        label_statements.append({
            "statement": f"MATCH (e) WHERE e.nodeKind='ontology_entity' AND e.id IN $entity_ids REMOVE e:{label}",
            "parameters": {"entity_ids": entity_ids},
        })
    for label, rows in sorted(grouped.items()):
        label_statements.append(_statement(
            f"UNWIND $rows AS row MATCH (e {{id:row.id}}) WHERE e.nodeKind='ontology_entity' "
            f"SET e:{label}, e.nodeType=$label, e.name=$label, e.displayName=$label",
            rows,
        ) | {"parameters": {"rows": rows, "label": label}})
    if label_statements:
        _post(label_statements)
    return {"typed_entities": len(entity_ids), "ontology_labels": sorted(grouped)}


def validate_entities_against_ontology(entities: list[dict]) -> dict:
    """Reject unknown classes and duplicate real-world nodes before any graph write."""
    ontology = json_read(ONTOLOGY_PATH)
    allowed = set(ontology["nodes"])
    entity_ids: set[str] = set()
    canonical_keys: set[tuple[str, str]] = set()
    violations: list[str] = []
    for row in entities:
        entity_id = row.get("entity_id", "")
        label = row.get("label", "")
        name = row.get("canonical_name", "")
        normalized_name = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
        if not entity_id:
            violations.append("entity is missing entity_id")
        elif entity_id in entity_ids:
            violations.append(f"duplicate entity_id {entity_id}")
        entity_ids.add(entity_id)
        if label not in allowed or not SAFE_LABEL.fullmatch(label):
            violations.append(f"{entity_id or '<missing id>'}: unknown ontology class {label or '<missing>'}")
        elif label in EMBEDDED_PROVENANCE_CLASSES:
            violations.append(f"{entity_id or '<missing id>'}: {label} is embedded provenance, not a canonical instance node")
        key = (label, normalized_name)
        if not normalized_name:
            violations.append(f"{entity_id or '<missing id>'}: missing canonical name")
        elif label not in SCOPED_LABELS and key in canonical_keys:
            violations.append(f"duplicate canonical node {label}:{name}")
        if label not in SCOPED_LABELS:
            canonical_keys.add(key)
    if violations:
        raise ValueError("Ontology entity validation failed: " + "; ".join(violations))
    return {"validated": len(entities), "violations": 0, "ontology_classes": sorted({row["label"] for row in entities})}


def build_entity_rows(run_id: str, document: dict, entities: list[dict], mentions: list[dict]) -> list[dict]:
    """Aggregate mentions into canonical nodes with exact parsed-source provenance."""
    mention_index = {row["mention_id"]: row for row in mentions}
    if len(mention_index) != len(mentions):
        raise ValueError("Mention aggregation failed: duplicate mention IDs were found.")
    claimed_mentions: set[str] = set()
    rows: list[dict] = []
    for entity in entities:
        records: list[str] = []
        mention_ids: list[str] = []
        document_ids: list[str] = []
        surface_texts: list[str] = []
        source_kinds: list[str] = []
        source_locators: list[str] = []
        sentence_ids: list[str] = []
        evidence_quotes: list[str] = []
        sections: list[str] = []
        page_refs: list[str] = []
        for mention_id in entity.get("mention_ids", []):
            if mention_id in claimed_mentions:
                raise ValueError(f"Mention aggregation failed: {mention_id} belongs to more than one canonical node.")
            mention = mention_index.get(mention_id)
            if mention is None:
                raise ValueError(f"Mention aggregation failed: {mention_id} is missing from mentions.jsonl.")
            if mention.get("label") != entity.get("label"):
                raise ValueError(
                    f"Mention aggregation failed: {mention_id} has class {mention.get('label')} but its node has class {entity.get('label')}."
                )
            claimed_mentions.add(mention_id)
            source = mention.get("source", {})
            context = source.get("context_sentences", [])
            source_kind = source.get("kind", "sentence_span" if source.get("sentence_id") else "")
            sentence_id = source.get("sentence_id") or (
                stable_id(
                    "metadata-sentence",
                    mention.get("document_id", document.get("id", "")),
                    source.get("field", "document.metadata"),
                )
                if source_kind == "document_metadata" else ""
            )
            source_locator = source.get("field", "") if source_kind == "document_metadata" else sentence_id
            if not source_kind or not source_locator:
                raise ValueError(
                    f"Mention aggregation failed: {mention_id} lacks an exact sentence or metadata source locator."
                )
            evidence_quote = source.get("evidence_quote", "")
            if not evidence_quote:
                raise ValueError(f"Mention aggregation failed: {mention_id} lacks an exact evidence quote.")
            record = {
                "mention_id": mention_id,
                "document_id": mention.get("document_id", document.get("id", "")),
                "surface_text": mention.get("surface_text", ""),
                "source_kind": source_kind,
                "source_locator": source_locator,
                "sentence_id": sentence_id,
                "sentence": evidence_quote,
                "evidence_quote": evidence_quote,
                "section": source.get("section_title", ""),
                "pages": source.get("pages", []),
                "context_sentences": context,
                "extraction_method": mention.get("extraction_method", ""),
                "attributes": mention.get("attributes", {}),
            }
            records.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            mention_ids.append(mention_id)
            document_ids.append(record["document_id"])
            surface_texts.append(record["surface_text"])
            source_kinds.append(record["source_kind"])
            source_locators.append(record["source_locator"])
            sentence_ids.append(record["sentence_id"])
            evidence_quotes.append(record["evidence_quote"])
            sections.append(record["section"])
            page_refs.append(", ".join(str(page) for page in record["pages"]))
        rows.append({
            "id": entity["entity_id"],
            "label": entity["label"],
            "name": entity.get("canonical_name", ""),
            "specific_content": (
                next((json.loads(record).get("evidence_quote", "") for record in records), entity.get("canonical_name", ""))
                if entity.get("label") in SCOPED_LABELS else entity.get("canonical_name", "")
            ),
            "identity_scope": entity.get("identity_scope", "occurrence" if entity.get("label") in SCOPED_LABELS else "canonical"),
            "attributes_json": json.dumps(entity.get("attributes", []), ensure_ascii=False, sort_keys=True),
            "aliases": entity.get("aliases", []),
            "run_id": run_id,
            "run_completed_at": now_iso(),
            "document_id": document.get("id", ""),
            "title": document.get("title", "") if entity["label"] == "Publication" else "",
            "filename": document.get("filename", "") if entity["label"] == "Publication" else "",
            "authors_text": document.get("authors_text", "") if entity["label"] == "Publication" else "",
            "mentions": records,
            "mention_ids": mention_ids,
            "mention_document_ids": document_ids,
            "mention_surface_texts": surface_texts,
            "mention_source_kinds": source_kinds,
            "mention_source_locators": source_locators,
            "mention_sentence_ids": sentence_ids,
            "mention_evidence_quotes": evidence_quotes,
            "mention_sections": sections,
            "mention_page_refs": page_refs,
        })
    unclaimed = sorted(set(mention_index) - claimed_mentions)
    if unclaimed:
        raise ValueError(f"Mention aggregation failed: {len(unclaimed)} accepted mentions are not attached to a canonical node.")
    return rows


def validate_relationships_against_ontology(entities: list[dict], relationships: list[dict]) -> dict:
    """Reject predicates and endpoints that violate the ontology domain/range."""
    ontology = json_read(ONTOLOGY_PATH)
    entity_classes = {row["entity_id"]: row["label"] for row in entities}
    violations: list[str] = []
    for row in relationships:
        predicate = row.get("predicate", "")
        spec = ontology["relationships"].get(predicate)
        subject_id = row.get("subject_entity_id", "")
        object_id = row.get("object_entity_id", "")
        subject_class = entity_classes.get(subject_id)
        object_class = entity_classes.get(object_id)
        if spec is None:
            violations.append(f"{predicate or '<missing predicate>'}: not defined by ontology")
            continue
        if subject_class is None or object_class is None:
            violations.append(f"{predicate}: endpoint is not a canonical entity in this run")
            continue
        if subject_class not in spec.get("domain", []):
            violations.append(f"{predicate}: subject class {subject_class} is outside domain")
        if object_class not in spec.get("range", []):
            violations.append(f"{predicate}: object class {object_class} is outside range")
    if violations:
        raise ValueError("Ontology relationship validation failed: " + "; ".join(violations))
    return {"validated": len(relationships), "violations": 0}


def graph_summary() -> dict:
    """Return health counts for the paper-only semantic graph."""
    results = _post([
        _statement(
            "MATCH (c:OntologyClass) WITH count(c) AS classes "
            "OPTIONAL MATCH (:OntologyClass)-[r]->(:OntologyClass) WHERE r.schemaEdge=true "
            "RETURN classes,count(r) AS schemaEdges"
        ),
        _statement(
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "RETURN coalesce(e.ontologyClass,'Unclassified') AS class,count(e) AS count "
            "ORDER BY count DESC,class"
        ),
        _statement(
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "RETURN sum(coalesce(e.mentionCount,0)) AS mentions, "
            "sum(coalesce(e.groundedMentionCount,0)) AS groundedMentions"
        ),
        _statement(
            "MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' AND b.nodeKind='ontology_entity' "
            "RETURN count(r) AS semanticRelationships"
        ),
        _statement("MATCH (n) WHERE n.confidence IS NOT NULL RETURN count(n) AS nodesWithConfidence"),
        _statement(
            "MATCH (e) WHERE e.nodeKind='ontology_entity' AND e.latestRunId IS NOT NULL "
            "RETURN e.latestRunId AS id,e.latestRunCompletedAt AS completedAt "
            "ORDER BY e.latestRunCompletedAt DESC LIMIT 1"
        ),
        _statement(
            "MATCH (n) WHERE n:ExtractedMention OR n:EvidenceFragment OR n:ExtractedAssertion OR "
            "n:ExtractionRun OR n:SourceDocument OR n:CanonicalEntity RETURN count(n) AS operationalNodes"
        ),
        _statement(
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "WITH e.ontologyClass AS class,toLower(trim(e.canonicalName)) AS name,"
            "coalesce(e.identityScope,'canonical') AS scope,count(e) AS copies "
            "WHERE copies>1 AND scope='canonical' RETURN count(*) AS duplicateCanonicalKeys"
        ),
    ])["results"]
    foundation = results[0]["data"][0]["row"] if results[0]["data"] else [0, 0]
    class_counts = [{"class": row["row"][0], "count": row["row"][1]} for row in results[1]["data"]]
    mention_counts = results[2]["data"][0]["row"] if results[2]["data"] else [0, 0]
    latest = results[5]["data"][0]["row"] if results[5]["data"] else ["", ""]
    return {
        "ok": True,
        "ontology_classes": foundation[0],
        "schema_edges": foundation[1],
        "materialized_ontology_nodes": foundation[0],
        "canonical_entities": sum(row["count"] for row in class_counts),
        "class_counts": class_counts,
        "mentions": mention_counts[0],
        "grounded_mentions": mention_counts[1],
        "semantic_relationships": results[3]["data"][0]["row"][0],
        "nodes_with_confidence": results[4]["data"][0]["row"][0],
        "operational_instance_nodes": results[6]["data"][0]["row"][0],
        "duplicate_canonical_keys": results[7]["data"][0]["row"][0],
        "latest_run": latest[0],
        "latest_run_completed_at": latest[1],
    }


def refresh_display_properties() -> None:
    """Make the ontology type visible while retaining extracted content separately."""
    _post([
        _statement(
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "SET e.name=e.ontologyClass, e.displayName=e.ontologyClass, e.nodeType=e.ontologyClass "
            "REMOVE e.confidence, e.score"
        ),
        _statement(
            "MATCH (m:ExtractedMention) OPTIONAL MATCH (m)-[:REFERS_TO]->(e:CanonicalEntity) "
            "WITH m,e, CASE "
            "WHEN toLower(coalesce(m.surfaceText,'')) IN ['it','they','them','this model','the model','this system','the system','this method','the method','this approach','the approach','these results'] "
            "AND e IS NOT NULL THEN m.surfaceText + ' → ' + e.name "
            "ELSE coalesce(m.surfaceText,e.name,m.id) END AS caption "
            "SET m.name=caption, m.displayName=caption, m.canonicalName=coalesce(e.name,m.canonicalName) "
            "REMOVE m.confidence"
        ),
        _statement(
            "MATCH (a:ExtractedAssertion) "
            "SET a.name=coalesce(a.predicate,a.id), a.displayName=coalesce(a.predicate,a.id) "
            "REMOVE a.confidence"
        ),
        _statement(
            "MATCH (f:EvidenceFragment) "
            "SET f.name=left(replace(coalesce(f.text,f.id),'\\n',' '),120), "
            "f.displayName=left(replace(coalesce(f.text,f.id),'\\n',' '),120)"
        ),
        _statement(
            "MATCH (e:CanonicalEntity) SET e.name=coalesce(e.canonicalName,e.name,e.id), "
            "e.displayName=coalesce(e.canonicalName,e.name,e.id)"
        ),
        _statement(
            "MATCH (d:SourceDocument) SET d.name=coalesce(d.title,d.filename,d.id), "
            "d.displayName=coalesce(d.title,d.filename,d.id)"
        ),
        _statement(
            "MATCH (run:ExtractionRun) SET run.name=coalesce(run.sourceFilename,run.id), "
            "run.displayName=coalesce(run.sourceFilename,run.id)"
        ),
    ])


def align_reusable_entity_ids(
    entities: list[dict], relationships: list[dict], document_id: str,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Reuse an existing graph ID for the same canonical ontology identity across papers."""
    response = _post([_statement(
        "MATCH (e) WHERE e.nodeKind='ontology_entity' "
        "AND coalesce(e.identityScope,'canonical')='canonical' "
        "RETURN e.ontologyClass,toLower(trim(e.canonicalName)),e.id,coalesce(e.sourceDocumentIds,[])"
    )])
    existing: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in response.get("results", [{}])[0].get("data", []):
        label, name, entity_id, source_documents = item["row"]
        existing[(label, name)].append({"id": entity_id, "documents": source_documents or []})

    remap: dict[str, str] = {}
    aligned_entities: list[dict] = []
    for entity in entities:
        original_id = entity["entity_id"]
        if entity["label"] in SCOPED_LABELS or entity.get("identity_scope") == "occurrence":
            aligned_entities.append(dict(entity))
            continue
        key = (entity["label"], entity.get("canonical_name", "").strip().casefold())
        candidates = existing.get(key, [])
        if candidates:
            preferred = min(
                candidates,
                key=lambda row: (
                    bool(row["documents"]) and all(value == document_id for value in row["documents"]),
                    -len(row["documents"]),
                    row["id"],
                ),
            )
            remap[original_id] = preferred["id"]
            aligned_entities.append({**entity, "entity_id": preferred["id"]})
        else:
            aligned_entities.append(dict(entity))

    aligned_relationships = [
        {
            **relationship,
            "subject_entity_id": remap.get(relationship["subject_entity_id"], relationship["subject_entity_id"]),
            "object_entity_id": remap.get(relationship["object_entity_id"], relationship["object_entity_id"]),
        }
        for relationship in relationships
    ]
    return aligned_entities, aligned_relationships, remap


def upsert_run(run_dir, parsed: dict, entities: list[dict], relationships: list[dict]) -> dict:
    """Write a paper-only graph; use the ontology solely for validation."""
    ontology = json_read(ONTOLOGY_PATH)
    ontology_spec = {"node_types": len(ontology["nodes"]), "predicate_types": len(ontology["relationships"])}
    document = parsed["document"]
    entities, relationships, _ = align_reusable_entity_ids(entities, relationships, document["id"])
    entity_validation = validate_entities_against_ontology(entities)
    relationship_validation = validate_relationships_against_ontology(entities, relationships)
    mentions = jsonl_read(run_dir / "mentions.jsonl")
    assertions = jsonl_read(run_dir / "assertions.jsonl")
    run_id = run_dir.name
    entity_rows = build_entity_rows(run_id, document, entities, mentions)

    # One-time migration away from ontology/schema and provenance-node representations.
    # The database contains only entities and semantic edges extracted from the paper.
    _post([
        _statement("MATCH (n:OntologyClass) DETACH DELETE n"),
        _statement(
            "MATCH (e:CanonicalEntity) SET e.nodeKind='ontology_entity' "
            "REMOVE e:CanonicalEntity:SourceDocument"
        ),
        _statement("MATCH (n:ExtractedMention) DETACH DELETE n"),
        _statement("MATCH (n:EvidenceFragment) DETACH DELETE n"),
        _statement("MATCH (n:ExtractedAssertion) DETACH DELETE n"),
        _statement("MATCH (n:ExtractionRun) DETACH DELETE n"),
        _statement("MATCH (n:SourceDocument) DETACH DELETE n"),
    ])
    _post([
        _statement("DROP CONSTRAINT canonical_entity_id IF EXISTS"),
        _statement("DROP CONSTRAINT extracted_mention_id IF EXISTS"),
        _statement("DROP CONSTRAINT evidence_fragment_id IF EXISTS"),
        _statement("DROP CONSTRAINT extracted_assertion_id IF EXISTS"),
        _statement("DROP CONSTRAINT extraction_run_id IF EXISTS"),
        _statement("DROP CONSTRAINT source_document_id IF EXISTS"),
    ])

    # Remove this paper's old mention slices from reusable entities that still
    # belong to other papers but are no longer in the current extraction. The
    # former delete-only cleanup left stale sourceDocumentIds on shared nodes.
    _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "AND NOT e.id IN $current_entity_ids "
            "AND any(source IN coalesce(e.sourceDocumentIds,[]) WHERE source<>$document_id) "
            "WITH e,[i IN range(0,size(coalesce(e.mentionDocumentIds,[]))-1) "
            "WHERE e.mentionDocumentIds[i]<>$document_id] AS keep, "
            "e.latestDocumentId=$document_id AS removingLatest "
            "SET e.mentions=[i IN keep | e.mentions[i]], "
            "e.mentionIds=[i IN keep | e.mentionIds[i]], "
            "e.mentionDocumentIds=[i IN keep | e.mentionDocumentIds[i]], "
            "e.mentionSurfaceTexts=[i IN keep | e.mentionSurfaceTexts[i]], "
            "e.mentionSourceKinds=[i IN keep | coalesce(e.mentionSourceKinds,[])[i]], "
            "e.mentionSourceLocators=[i IN keep | coalesce(e.mentionSourceLocators,[])[i]], "
            "e.mentionSentenceIds=[i IN keep | e.mentionSentenceIds[i]], "
            "e.mentionEvidenceQuotes=[i IN keep | e.mentionEvidenceQuotes[i]], "
            "e.mentionSections=[i IN keep | e.mentionSections[i]], "
            "e.mentionPageRefs=[i IN keep | e.mentionPageRefs[i]] "
            "WITH e "
            "SET e.mentionCount=size(e.mentions), "
            "e.groundedMentionCount=size([quote IN e.mentionEvidenceQuotes WHERE trim(quote)<>'' ]), "
            "e.surfaceForms=reduce(acc=[],x IN e.mentionSurfaceTexts | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.sourceDocumentIds=reduce(acc=[],x IN e.mentionDocumentIds | CASE WHEN x IN acc THEN acc ELSE acc+x END)"
        ),
        "parameters": {
            "document_id": document["id"],
            "current_entity_ids": [row["id"] for row in entity_rows],
        },
    }])

    # Remove nodes from an older extraction of this same paper when they are no
    # longer present in the latest canonical entity set. Cross-paper entities survive.
    _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "AND NOT e.id IN $current_entity_ids "
            "AND all(source IN coalesce(e.sourceDocumentIds,[]) WHERE source=$document_id) "
            "DETACH DELETE e"
        ),
        "parameters": {
            "document_id": document["id"],
            "current_entity_ids": [row["id"] for row in entity_rows],
        },
    }])

    _post([{
        "statement": (
            "UNWIND $rows AS row "
            "MERGE (e {id:row.id, nodeKind:'ontology_entity'}) "
            "WITH e,row,CASE WHEN size(coalesce(e.mentions,[]))=size(coalesce(e.mentionDocumentIds,[])) "
            "THEN [i IN range(0,size(coalesce(e.mentions,[]))-1) WHERE e.mentionDocumentIds[i]<>row.document_id] "
            "ELSE [] END AS keep "
            "SET e.canonicalName=row.name,e.ontologyClass=row.label,e.nodeType=row.label, "
            "e.name=row.label,e.displayName=row.label,e.extractedValue=row.name,e.specificContent=row.specific_content, "
            "e.identityScope=row.identity_scope,e.attributesJson=row.attributes_json, "
            "e.aliases=reduce(acc=[],x IN coalesce(e.aliases,[])+row.aliases | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.mentions=[i IN keep | e.mentions[i]]+row.mentions, "
            "e.mentionIds=[i IN keep | e.mentionIds[i]]+row.mention_ids, "
            "e.mentionDocumentIds=[i IN keep | e.mentionDocumentIds[i]]+row.mention_document_ids, "
            "e.mentionSurfaceTexts=[i IN keep | e.mentionSurfaceTexts[i]]+row.mention_surface_texts, "
            "e.mentionSourceKinds=[i IN keep | coalesce(e.mentionSourceKinds,[])[i]]+row.mention_source_kinds, "
            "e.mentionSourceLocators=[i IN keep | coalesce(e.mentionSourceLocators,[])[i]]+row.mention_source_locators, "
            "e.mentionSentenceIds=[i IN keep | e.mentionSentenceIds[i]]+row.mention_sentence_ids, "
            "e.mentionEvidenceQuotes=[i IN keep | e.mentionEvidenceQuotes[i]]+row.mention_evidence_quotes, "
            "e.mentionSections=[i IN keep | e.mentionSections[i]]+row.mention_sections, "
            "e.mentionPageRefs=[i IN keep | e.mentionPageRefs[i]]+row.mention_page_refs, "
            "e.latestRunId=row.run_id,e.latestRunCompletedAt=row.run_completed_at,e.latestDocumentId=row.document_id, "
            "e.latestDocumentMentionCount=size(row.mentions),e.updatedAt=datetime() "
            "FOREACH (_ IN CASE WHEN row.label='Publication' THEN [1] ELSE [] END | "
            "SET e.title=row.title,e.filename=row.filename,e.authorsText=row.authors_text,e.documentId=row.document_id) "
            "WITH e,row "
            "SET e.mentionCount=size(e.mentions), "
            "e.groundedMentionCount=size([quote IN e.mentionEvidenceQuotes WHERE trim(quote)<>'']), "
            "e.surfaceForms=reduce(acc=[],x IN e.mentionSurfaceTexts | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.sourceDocumentIds=reduce(acc=[],x IN e.mentionDocumentIds | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.latestDocumentGroundedMentionCount=size([quote IN row.mention_evidence_quotes WHERE trim(quote)<>'']) "
            "REMOVE e.confidence, e.score"
        ),
        "parameters": {"rows": entity_rows},
    }])

    ontology_labels = apply_ontology_labels(entity_rows)
    constraint_statements = []
    for label in ontology_labels["ontology_labels"]:
        constraint_name = "ontology_entity_" + re.sub(r"[^a-z0-9]+", "_", label.casefold()) + "_id"
        constraint_statements.append(_statement(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        ))
    if constraint_statements:
        _post(constraint_statements)

    # Only evidence-backed, ontology-validated paper relationships are written.
    _post([{
        "statement": (
            "MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' AND b.nodeKind='ontology_entity' "
            "AND (r.documentId=$document_id OR "
            "(r.documentId IS NULL AND $document_id IN coalesce(a.sourceDocumentIds,[]) "
            "AND $document_id IN coalesce(b.sourceDocumentIds,[]))) DELETE r"
        ),
        "parameters": {"document_id": document["id"]},
    }])
    assertion_by_id = {row["assertion_id"]: row for row in assertions}
    by_predicate: dict[str, list[dict]] = defaultdict(list)
    for relationship in relationships:
        predicate = relationship["predicate"]
        if not SAFE_RELATIONSHIP.fullmatch(predicate):
            raise ValueError(f"Unsafe Neo4j relationship type: {predicate}")
        evidence = [assertion_by_id[item] for item in relationship.get("assertion_ids", []) if item in assertion_by_id]
        by_predicate[predicate].append({
            "id": relationship["relationship_id"],
            "subject_id": relationship["subject_entity_id"],
            "object_id": relationship["object_entity_id"],
            "support_count": int(relationship.get("support_count", len(evidence))),
            "assertion_ids": relationship.get("assertion_ids", []),
            "run_id": run_id,
            "document_id": document["id"],
            "evidence_quotes": [row.get("evidence_quote") or "" for row in evidence[:10]],
            "evidence_sentence_ids": [(row.get("evidence_sentence_ids") or [""])[0] for row in evidence[:10]],
            "evidence_sections": [row.get("section_id") or "" for row in evidence[:10]],
            "evidence_page_refs": [", ".join(str(page) for page in row.get("pages", [])) for row in evidence[:10]],
            "trigger_texts": [row.get("predicate_text", "") or "" for row in evidence[:10]],
        })
    for predicate, rows in by_predicate.items():
        _post([_statement(
            f"UNWIND $rows AS row MATCH (s {{id:row.subject_id}}),(o {{id:row.object_id}}) "
            f"WHERE s.nodeKind='ontology_entity' AND o.nodeKind='ontology_entity' "
            f"MERGE (s)-[r:{predicate} {{relationshipId:row.id}}]->(o) "
            "SET r.supportCount=row.support_count,r.assertionIds=row.assertion_ids,r.runId=row.run_id,r.documentId=row.document_id,"
            "r.evidenceQuotes=row.evidence_quotes,r.evidenceSentenceIds=row.evidence_sentence_ids,"
            "r.evidenceSections=row.evidence_sections,r.evidencePageRefs=row.evidence_page_refs,"
            "r.triggerTexts=row.trigger_texts,r.traceable=true",
            rows,
        )])

    refresh_display_properties()
    verification = _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' AND e.latestRunId=$run_id "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "WITH count(DISTINCT e) AS entities,"
            "count(DISTINCT CASE WHEN size(labels(e))=1 AND labels(e)[0]=e.ontologyClass THEN e END) AS typedEntities, "
            "sum(e.latestDocumentMentionCount) AS mentions,sum(e.latestDocumentGroundedMentionCount) AS grounded "
            "CALL () { MATCH (schema:OntologyClass) RETURN count(schema) AS materializedOntologyNodes } "
            "CALL () { MATCH (n) WHERE n:ExtractedMention OR n:EvidenceFragment OR n:ExtractedAssertion OR "
            "n:ExtractionRun OR n:SourceDocument OR n:CanonicalEntity RETURN count(n) AS operationalNodes } "
            "CALL () { MATCH (node) WHERE node.nodeKind='ontology_entity' "
            "WITH node.ontologyClass AS class,toLower(trim(node.canonicalName)) AS name,count(node) AS copies "
            "WHERE copies>1 AND NOT class IN $scoped_labels RETURN count(*) AS duplicateCanonicalKeys } "
            "CALL () { MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' AND b.nodeKind='ontology_entity' "
            "AND r.runId=$run_id RETURN count(r) AS semanticRelationships,"
            "count(CASE WHEN r.traceable=true AND size(coalesce(r.evidenceQuotes,[]))>0 THEN 1 END) AS traceableRelationships } "
            "RETURN entities,typedEntities,mentions,grounded,operationalNodes,duplicateCanonicalKeys,"
            "materializedOntologyNodes,semanticRelationships,traceableRelationships"
        ),
        "parameters": {
            "run_id": run_id,
            "document_id": document["id"],
            "scoped_labels": sorted(SCOPED_LABELS),
        },
    }])
    values = verification["results"][0]["data"][0]["row"]
    if values[0] != len(entities) or values[1] != len(entities):
        raise RuntimeError(f"Neo4j ontology alignment failed: expected {len(entities)} entities, found {values[0]}/{values[1]} typed.")
    if values[2] != len(mentions) or values[3] != len(mentions):
        raise RuntimeError(f"Neo4j mention aggregation failed: expected {len(mentions)} grounded mentions, found {values[2]}/{values[3]}.")
    if values[4] or values[5] or values[6]:
        raise RuntimeError(
            f"Neo4j paper-only invariant failed: {values[4]} operational nodes, "
            f"{values[5]} duplicate canonical keys, and {values[6]} materialized ontology nodes."
        )
    if values[7] != len(relationships) or values[8] != len(relationships):
        raise RuntimeError(
            f"Neo4j relationship traceability failed: expected {len(relationships)}, "
            f"found {values[7]} semantic and {values[8]} traceable relationships."
        )

    return {
        "status": "upserted",
        "database": "neo4j",
        "graph_model": "paper_semantic_graph_v3",
        "http_endpoint": "http://localhost:7477/",
        "browser": "http://127.0.0.1:7477/",
        "run_id": run_id,
        "document_id": document["id"],
        "entities": values[0],
        "mentions": values[2],
        "mention_nodes": 0,
        "evidence_fragment_nodes": 0,
        "operational_instance_nodes": values[4],
        "duplicate_canonical_keys": values[5],
        "assertions": len(assertions),
        "relationships": values[7],
        "traceable_relationships": values[8],
        "materialized_ontology_nodes": values[6],
        "ontology_spec_used_for_validation": ontology_spec,
        "ontology_alignment": {"paper_entities": values[0], "correctly_labeled_entities": values[1]},
        "ontology_labels": ontology_labels,
        "entity_validation": entity_validation,
        "relationship_validation": relationship_validation,
        "traceability": {
            "mentions": values[2],
            "mentions_with_exact_evidence": values[3],
            "storage": "embedded JSON records on canonical ontology nodes",
        },
    }


def remove_document(document_id: str, run_id: str = "") -> dict:
    """Remove one paper's graph contribution without deleting entities shared by other papers."""
    if not document_id or not str(document_id).strip():
        raise ValueError("A document ID is required to remove a paper from Neo4j.")

    before = _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "WITH count(DISTINCT e) AS entities "
            "CALL () { MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' "
            "AND b.nodeKind='ontology_entity' AND r.documentId=$document_id "
            "RETURN count(r) AS relationships } "
            "RETURN entities,relationships"
        ),
        "parameters": {"document_id": document_id},
    }])
    values = before["results"][0]["data"][0]["row"] if before["results"][0]["data"] else [0, 0]

    _post([{
        "statement": (
            "MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' "
            "AND b.nodeKind='ontology_entity' AND r.documentId=$document_id DELETE r"
        ),
        "parameters": {"document_id": document_id},
    }])

    # Shared canonical identities survive. Only the mention/evidence slices
    # contributed by this document are removed from their embedded arrays.
    _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "AND any(source IN coalesce(e.sourceDocumentIds,[]) WHERE source<>$document_id) "
            "WITH e,[i IN range(0,size(coalesce(e.mentionDocumentIds,[]))-1) "
            "WHERE e.mentionDocumentIds[i]<>$document_id] AS keep, "
            "e.latestDocumentId=$document_id AS removingLatest "
            "SET e.mentions=[i IN keep | e.mentions[i]], "
            "e.mentionIds=[i IN keep | e.mentionIds[i]], "
            "e.mentionDocumentIds=[i IN keep | e.mentionDocumentIds[i]], "
            "e.mentionSurfaceTexts=[i IN keep | e.mentionSurfaceTexts[i]], "
            "e.mentionSourceKinds=[i IN keep | coalesce(e.mentionSourceKinds,[])[i]], "
            "e.mentionSourceLocators=[i IN keep | coalesce(e.mentionSourceLocators,[])[i]], "
            "e.mentionSentenceIds=[i IN keep | e.mentionSentenceIds[i]], "
            "e.mentionEvidenceQuotes=[i IN keep | e.mentionEvidenceQuotes[i]], "
            "e.mentionSections=[i IN keep | e.mentionSections[i]], "
            "e.mentionPageRefs=[i IN keep | e.mentionPageRefs[i]] "
            "WITH e,removingLatest SET e.mentionCount=size(e.mentions), "
            "e.groundedMentionCount=size([quote IN e.mentionEvidenceQuotes WHERE trim(quote)<>'' ]), "
            "e.surfaceForms=reduce(acc=[],x IN e.mentionSurfaceTexts | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.sourceDocumentIds=reduce(acc=[],x IN e.mentionDocumentIds | CASE WHEN x IN acc THEN acc ELSE acc+x END), "
            "e.latestDocumentId=CASE WHEN removingLatest THEN "
            "coalesce(e.mentionDocumentIds[-1],'') ELSE e.latestDocumentId END, "
            "e.latestRunId=CASE WHEN removingLatest THEN '' ELSE e.latestRunId END, "
            "e.updatedAt=datetime()"
        ),
        "parameters": {"document_id": document_id},
    }])

    # Nodes supported only by the removed paper have no remaining provenance
    # and are therefore deleted with their remaining incident relationships.
    _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "AND all(source IN coalesce(e.sourceDocumentIds,[]) WHERE source=$document_id) "
            "DETACH DELETE e"
        ),
        "parameters": {"document_id": document_id},
    }])
    refresh_display_properties()

    verification = _post([{
        "statement": (
            "MATCH (e) WHERE e.nodeKind='ontology_entity' "
            "AND $document_id IN coalesce(e.sourceDocumentIds,[]) "
            "WITH count(e) AS entities "
            "CALL () { MATCH (a)-[r]->(b) WHERE a.nodeKind='ontology_entity' "
            "AND b.nodeKind='ontology_entity' AND r.documentId=$document_id "
            "RETURN count(r) AS relationships } "
            "RETURN entities,relationships"
        ),
        "parameters": {"document_id": document_id},
    }])
    remaining = verification["results"][0]["data"][0]["row"] if verification["results"][0]["data"] else [0, 0]
    if remaining != [0, 0]:
        raise RuntimeError(
            f"Neo4j paper removal verification failed: {remaining[0]} entities and "
            f"{remaining[1]} relationships still reference {document_id}."
        )
    return {
        "status": "removed",
        "database": "neo4j",
        "run_id": run_id,
        "document_id": document_id,
        "entities_removed_or_detached": values[0],
        "relationships_removed": values[1],
        "browser": "http://127.0.0.1:7477/",
    }


def reset_graph() -> dict:
    """Delete every node and relationship from the configured Neo4j database."""
    before = _post([
        _statement("MATCH (n) RETURN count(n) AS nodes"),
        _statement("MATCH ()-[r]->() RETURN count(r) AS relationships"),
    ])["results"]
    nodes = before[0]["data"][0]["row"][0] if before[0]["data"] else 0
    relationships = before[1]["data"][0]["row"][0] if before[1]["data"] else 0

    _post([_statement("MATCH (n) DETACH DELETE n")])

    after = _post([
        _statement("MATCH (n) RETURN count(n) AS nodes"),
        _statement("MATCH ()-[r]->() RETURN count(r) AS relationships"),
    ])["results"]
    remaining_nodes = after[0]["data"][0]["row"][0] if after[0]["data"] else 0
    remaining_relationships = after[1]["data"][0]["row"][0] if after[1]["data"] else 0
    if remaining_nodes or remaining_relationships:
        raise RuntimeError(
            "Neo4j reset verification failed: "
            f"{remaining_nodes} nodes and {remaining_relationships} relationships remain."
        )
    return {
        "status": "reset",
        "database": "neo4j",
        "nodes_removed": nodes,
        "relationships_removed": relationships,
        "nodes_remaining": remaining_nodes,
        "relationships_remaining": remaining_relationships,
        "browser": "http://127.0.0.1:7477/",
    }
