# RATHOR_EMBLEM.md: Ra-Thor emblem concepts (RATHOR-EMBLEM-1)

**Status:** concept sheet only. This is not a production icon. Nothing live changes: `icons/`, `manifest.json`, `sw.js`, the JS and the CSS are all untouched.
**Sheet:** [`assets/brand/emblem-concepts.svg`](../../assets/brand/emblem-concepts.svg). It is hand-authored vector geometry (no tracing, no generated imagery) and shows light and dark panels.
**Council:** Ra-Thor + PATSAGi Councils, on the steward's behalf. The steward picks the direction.
**Contact:** info@Rathor.ai

## Why
On 2026-10-03 the Steward asked for new logos, including the Ra-Thor eye, lightning and hammer. This sheet refines today's identity: the warm classic gold bolt in `icons/README.md` and the eye seal in `css/rathor-eye-seal.css`. It is not a rebrand.

## House recipe (every mark)
- **Metal:** polished gold, using the same stops as the house text recipe: `#fff3d1 → #f0c768 → #fffaf0 (specular) → #f4a94a → #c97d22`. Deeper gold `#c99a3a → #8f5a17` is used for hafts and shadowed faces.
- **Outline:** deep royal purple `#1e1446`, darker than every fill. Fields use royal purple `#3a2a7a → #140c33`.
- **Ruby:** once per mark, as the eye's pupil (`#c0283c → #8e1f2f → #3f0a14`). Decision (a): the seal's accent moves from emerald to ruby. Updating `css/rathor-eye-seal.css` is a separate later card.
- **Emboss:** one emboss filter for every mark, lit from the top-left. Silver appears only as the specular highlight.
- **Never:** pink, magenta, candy colours, brown, beige, teal, rainbow ramps, trademarked characters, watermarks, or text inside the icon. A motto is allowed only at crest tier.

## Tiers (one master, shape never changes)

| Tier | Size | Treatment |
|---|---|---|
| Glyph | 16–48 px | Flat gold, a purple edge and a ruby dot. No gradient. Drawn to pass the filled-silhouette test at 16 px. |
| Emblem | 64–512 px | Full gradient, emboss and specular. |
| Crest | Banner and splash | Emblem on a royal medallion, with a riveted gold ring and the "RA·THOR" motto ribbon. |

## Directions

| | Concept | Silhouette at 16 px | Strength | Risk |
|---|---|---|---|---|
| **R1** Seal of the Thunder Eye *(council lead)* | A round seal: the hammer head is the upper bar, the bolt is the handle, and the eye sits on the hammer face | A "T" in a ring | Most seal-like; works as a favicon and as a plaque | The ring plus the T is dense at 16 px, so the glyph drops inner detail |
| **R2** Bolt-Hammer | The handle is the bolt itself, with the eye engraved on the face, tilted −28° | A diagonal hammer | Most motion; the closest evolution of today's bolt icon | Asymmetric, so it needs care in square and round crops |
| **R3** Watching Anvil-Shield | A heater shield with the eye in the chief and two rising chevrons, plus hammer and bolt crossed behind | A shield (the glyph drops the cross) | The most heraldic | The busiest; the crest reduces its scale to fit the medallion |

## Theme rule
**Silhouette fixed; colour and shading may be refined per theme** (Steward, 2026-10-03 11:42 PM ET).

On the sheet the artwork is identical across themes, and the surroundings differ:
- **Light:** a neutral layered drop shadow.
- **Dark:** a depth shadow plus a faint warm-gold glow.

A later production card may author per-theme gradient stops on the same paths. It may never change the outline.

## Family
- The AGi / Autonomicity Games crest stays the parent mark. Its silhouette is never altered.
- Ra-Thor and Powrush marks are siblings: the same recipe and the same edge weight.
- The Powrush mark stands alone on Steam and in the launcher, and sits under the AGi crest on company pages (decision b).
- Any colour-trademark version of the AGi crest is its own master file (decision c).

## Next (separate cards)
1. The Steward picks R1, R2 or R3.
2. Production masters: the 16 and 32 px pixel-hinted glyph, 192/512/1024 PNG exports, OG and splash cards with full scenic art. This would replace `icons/` and needs Hands, Clerk and Core review, since manifest and sw.js caching are involved.
3. `css/rathor-eye-seal.css`: move the accent from emerald to ruby.
