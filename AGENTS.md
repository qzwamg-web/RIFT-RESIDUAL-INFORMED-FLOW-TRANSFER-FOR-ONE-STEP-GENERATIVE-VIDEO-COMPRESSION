# RIFT

Read README.md and the architecture, training and evaluation guides before changes.
Use scripts/final.py as the common command-line entry point.
Preserve checkpoint compatibility and the shared training/inference restoration
pipeline. Do not change tensor keys or numerical operations as part of documentation
maintenance. Keep model weights, datasets, credentials and generated runs out of Git.
Maintain the documented 96-frame, GOP25 evaluation protocol. External reference
curves retain their source protocols and must not be presented as local reproductions.
