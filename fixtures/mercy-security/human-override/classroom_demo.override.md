# Companion override note — GE-GAP-HUMAN-OVERRIDE

This note is **not** an unattended admit. The paired ingest
`blocked/human_override_classroom_demo.txt` still BLOCKS on
`IngestionScanner::admit_or_block`.

A human steward, after review, may write one sidecar row with
`DecisionRecord::human_override` only:

- actor = human
- verdict = override
- prev_verdict = block
- non-empty rationale (this note)

Rationale used by the named lock:

Steward reviewed the blocked classroom demo ingest after the
unattended gate. Log override only. Do not skip admit_or_block.
Medium+ stays BLOCK for unattended ingest.

Empty rationale is refused. The log stores SHA-256 of the ingest,
never the raw paste.

This pair closes documentation completeness. It does not claim
override is safe, measure an override rate, or demonstrate Combined AGSi.
Inspect ≠ METR. Contact: info@Rathor.ai
