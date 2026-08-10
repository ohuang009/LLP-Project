# Local Neo4j

- `config/neo4j.conf` is the tracked configuration used at startup.
- `storage/` contains the graph database and Neo4j-generated runtime files; it is ignored by Git.

The Neo4j distribution is installed outside this repository. Startup, ports, backup, and path-update instructions are in `../STANDARD_OPERATING_PROCEDURE.md`.
