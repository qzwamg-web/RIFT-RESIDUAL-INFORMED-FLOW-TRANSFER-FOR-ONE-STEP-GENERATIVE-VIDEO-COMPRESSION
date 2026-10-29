# RIFT C512 source release

Use `scripts/final.py` for the preserved C512 endpoint. Read README.md and
`docs/FRAMEWORK.md`, `docs/TRAINING.md`, `docs/TESTING.md` before changes.
Keep weights/data/videos/credentials out of Git. Preserve active inherited
adapters and the original C512 checkpoint as the numerical anchor. The historical
96-frame GOP25 evaluation protocol is documented separately from newer protocols.
Do not claim a merged BF16 export is bitwise equivalent without an endpoint test.
