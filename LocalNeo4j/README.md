# Local Neo4j runtime

This folder supports the single database used by the current Engineered Water Systems workbench.

## Active instance

| Instance | HTTP | Bolt | Purpose |
|---|---|---|---|
| `engineered-water-ontology` | `http://localhost:7477/` | `bolt://localhost:7690` | Paper-derived ontology-typed nodes, relationships, and evidence |

Despite the historical instance name, the database contains paper-derived instance data only. The ontology in `../Pipeline/GENERAL/ontology/` is used as an extraction and validation contract and is not materialized as ontology-class nodes.

## Starting the local instance

Install Java 21 or 25 and extract Neo4j Community locally. Set `NEO4J_HOME` to that distribution
and point `NEO4J_CONF` at this instance before starting the server:

```bash
export NEO4J_CONF="$PWD/LocalNeo4j/instances/engineered-water-ontology/conf"
"$NEO4J_HOME/bin/neo4j-admin" server console
```

Stop the server with `Ctrl-C`. Run `python -m Pipeline.GENERAL.node_only_server` from the
repository root to launch the workbench.

## Repository boundary

The Neo4j distribution under `runtime/`, downloaded installers, graph assets, logs, and database
files under `instances/` are local state and are excluded by the repository `.gitignore`.

Older `ontology`, `one-paper`, `all-samples`, and `water-ontology` databases belonged to retired workflows and were moved to the cleanup backup during repository consolidation.
