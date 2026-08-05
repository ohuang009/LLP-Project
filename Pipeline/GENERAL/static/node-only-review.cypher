// 1. The paper knowledge graph: extracted nodes and extracted semantic edges only.
MATCH p=(source)-[relationship]->(target)
WHERE source.nodeKind='ontology_entity' AND target.nodeKind='ontology_entity'
RETURN p
LIMIT 150;

// 2. Every paper node. The circle caption is the ontology type; inspect
// canonicalName, extractedValue, and specificContent for the paper-specific value.
MATCH (node)
WHERE node.nodeKind='ontology_entity'
RETURN node
LIMIT 150;

// 3. Which models does the extracted AI system use?
MATCH (system:AISystem)-[relationship:USES_MODEL]->(model:Model)
RETURN system.canonicalName AS aiSystem,
       model.canonicalName AS model,
       relationship.evidenceQuotes AS evidence,
       relationship.evidenceSentenceIds AS sentenceIds;

// 4. What tasks, challenges, and measured results are connected to the AI system?
MATCH (system:AISystem {canonicalName:'WaterRAG'})-[relationship]-(answer)
WHERE answer:Task OR answer:Challenge OR answer:Observation OR answer:Metric
RETURN type(relationship) AS predicate,
       answer.ontologyClass AS nodeType,
       answer.specificContent AS specificContent,
       relationship.evidenceQuotes AS evidence
ORDER BY predicate, nodeType;

// 5. Quantitative observations remain occurrence-scoped and are never merged by value.
MATCH (observation:Observation)
RETURN observation.id,
       observation.specificContent,
       observation.attributesJson,
       observation.mentionSentenceIds,
       observation.mentionEvidenceQuotes
ORDER BY observation.mentionSentenceIds[0];

// 6. Inspect all mentions aggregated under one canonical identity node.
MATCH (entity:AISystem {canonicalName:'WaterRAG'})
UNWIND entity.mentions AS mention
RETURN entity.ontologyClass AS nodeType, entity.canonicalName AS specificNode, mention
LIMIT 150;

// 7. Every semantic edge must contain exact sentence evidence.
MATCH (source)-[relationship]->(target)
WHERE source.nodeKind='ontology_entity' AND target.nodeKind='ontology_entity'
RETURN source.canonicalName AS source,
       type(relationship) AS predicate,
       target.canonicalName AS target,
       relationship.supportCount AS supportCount,
       relationship.evidenceQuotes AS evidence,
       relationship.evidenceSentenceIds AS sentenceIds;

// 8. These counts must all be zero.
MATCH (node)
WHERE node:OntologyClass OR node:ExtractedMention OR node:EvidenceFragment OR
      node:ExtractedAssertion OR node:ExtractionRun OR node:SourceDocument OR node:CanonicalEntity
RETURN count(node) AS forbiddenGraphNodes;

// 9. Canonical identity nodes must not be duplicated. Occurrence-scoped statistics are allowed.
MATCH (entity)
WHERE entity.nodeKind='ontology_entity' AND entity.identityScope='canonical'
WITH entity.ontologyClass AS nodeType, toLower(trim(entity.canonicalName)) AS canonicalName, count(*) AS copies
WHERE copies > 1
RETURN nodeType, canonicalName, copies;

// 10. Confidence is deliberately absent from graph nodes.
MATCH (node)
WHERE node.confidence IS NOT NULL OR node.score IS NOT NULL
RETURN count(node) AS nodesWithConfidence;
