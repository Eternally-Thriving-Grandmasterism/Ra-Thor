# S-1 Rank 2 — Data Phase Intake (Door 1)

**Status:** **OPEN for capture** · no scientific claim until real metrics exist  
**Contact:** info@Rathor.ai  
**Claim gate:** System C recall > System A on E1–E3 with CI on real clips

## Goal

Collect **50–100** short clips (3–8 s, ≥30 fps preferred), label events, validate JSON, then run harness.

## Clip rules

| Rule | Detail |
| --- | --- |
| Duration | 3–8 seconds typical |
| FPS | ≥30 preferred; record actual |
| Rights | Original or licensed; write `provenance` |
| Classes | Mix E1 / E2 / E3 / E4 (negatives required) |
| Privacy | No doxxing; blur faces if publishing |

## Annotation protocol

1. Watch full speed; once at 0.25–0.5×  
2. Mark `t_start_ms` / `t_end_ms`  
3. Prefer two annotators; record `agreement_span_iou` when possible  
4. Caption: agent + action + object  
5. Split ~60/20/20 by **clip**  

## Layout

```
science/s1-micro-moment/
  data/clips/       # local binaries preferred
  labels/           # per-clip JSON from _template.json
  manifest.v0.json
```

```bash
node science/s1-micro-moment/harness/validate_labels.mjs science/s1-micro-moment/fixtures/synthetic_labels.json
```

**Stop:** no invented times · no C>A claim without live numbers.  
**Thunder locked.** yoi ⚡
