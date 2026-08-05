# Persistent water-domain lexicon

The lexicon will be a persistent, versioned registry of reviewed canonical nodes and aliases seen across papers.
It will be created when the extraction plan is approved.

It must not be an automatically growing list of every extracted surface phrase. A new entry becomes reusable only
after deterministic validation or human approval. This prevents one bad extraction from contaminating later papers.

The approved lexicon record will include at least:

- canonical ID and ontology class
- canonical name
- exact aliases and acronyms
- authoritative identifiers, when available
- definition or disambiguation note
- first-seen and supporting document IDs
- provenance quote and sentence ID for corpus-derived entries
- review status, reviewer, and lexicon version
- active/deprecated status and replacement ID

