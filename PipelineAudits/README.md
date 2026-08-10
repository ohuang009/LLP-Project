# Generated pipeline evidence

Every pipeline run writes a self-contained folder under `runs/`. These files are outputs, not source code and not hidden input to later runs.

The source repository ignores runs, benchmarks, exports, reports, and temporary audit data. They can be archived or deleted without changing extraction behavior, but deletion may erase useful evaluation history and copied source PDFs.
