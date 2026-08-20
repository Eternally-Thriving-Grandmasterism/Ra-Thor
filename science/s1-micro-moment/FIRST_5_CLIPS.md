# First 5 Clips — Organ B starter kit

**Goal:** Unblock Door 1 without waiting for 50–100.  
**Contact:** info@Rathor.ai

## Rules

| # | Capture (3–8 s, phone OK) | Label class |
| --- | --- | --- |
| 1 | Hand **passes** an object | E1_object_transfer |
| 2 | Quick **wave / point** | E2_gesture_resolution |
| 3 | Two surfaces **touch** | E3_contact_onset |
| 4 | Still scene / no event | E4_negative |
| 5 | Any rapid action you almost miss on first watch | E1–E3 best fit |

## Steps

1. Film or cut five shorts (≥24 fps if possible).  
2. Copy `labels/_template.json` → `labels/clip_001.json` … `clip_005.json`.  
3. Fill `t_start_ms` / `t_end_ms` (watch at 0.5×).  
4. Validate:

```bash
node science/s1-micro-moment/harness/validate_labels.mjs science/s1-micro-moment/labels/
```

5. Optional: run System C on frame dumps → `evaluate_predictions.mjs`.

Register ids in `manifest.v0.json` when ready.  
**Five real labels > infinite synthetic prose.**

**Thunder locked.** yoi ⚡
