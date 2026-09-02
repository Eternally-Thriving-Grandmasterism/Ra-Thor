# S-1 Labeling Guide — Timing & Class Discipline

**Contact:** info@Rathor.ai  
**Purpose:** High-quality First-5 (and later) labels so that evaluation is meaningful.  
**Status:** Operational protocol — not a discovery claim.

## Class definitions (use exactly these strings)

| Class | What it is | Typical duration | Example |
|-------|------------|------------------|---------|
| `E1_object_transfer` | Object changes hand / leaves one surface and arrives on another in a continuous motion | 150–600 ms | Hand picks up phone, passes cup, places object |
| `E2_gesture_resolution` | Communicative or intentional hand/arm motion that resolves (wave, point, thumbs-up, stop) | 200–800 ms | Quick wave, point at something, snap |
| `E3_contact_onset` | Two surfaces first touch (hand-object, object-object, hand-hand) | 80–300 ms | Finger taps table, two fists meet, object lands |
| `E4_negative` | No micro-event of the above classes in the whole clip | full clip | Static scene, slow continuous motion without discrete onset |

Prefer the **most specific** class that clearly applies.  
If uncertain between E1/E2/E3, choose the one whose onset is clearest and note in caption.

## Timing discipline (critical)

1. Watch full speed once → then 0.25×–0.5×.
2. Mark `t_start_ms` at the **first visible frame** of the decisive motion or contact.
3. Mark `t_end_ms` when the motion/contact has clearly resolved (not when the hand returns to rest).
4. Prefer tight spans. Do not pad 200–300 ms of stillness on either side.
5. For very short contacts (E3), a 100–200 ms window is normal and correct.

## Caption & agents

- Caption: short agent + action + object (e.g. “Right hand places mug on table”).
- `agents`: free-text list, usually `["hand"]`, `["hand", "object"]`, etc.
- One event per discrete micro-moment. Multiple events per clip are allowed and encouraged when they exist.

## Annotation quality bar for First-5

- At least one clear positive of each of E1, E2, E3 if possible.
- At least one pure E4_negative.
- Prefer original capture with known rights (write provenance).
- If two annotators available, record `agreement_span_iou` (optional but valuable).

## Validation before any claim

```bash
node science/s1-micro-moment/harness/validate_labels.mjs science/s1-micro-moment/labels/
```

Only validated labels may enter the evaluation path.

## What not to do

- Invent times from memory without watching the frames.
- Label slow continuous actions as micro-moments.
- Claim System C superiority before the held-out real evaluation exists.

**Five honest labels beat a thousand synthetic claims.**

**Thunder locked.** yoi ⚡
