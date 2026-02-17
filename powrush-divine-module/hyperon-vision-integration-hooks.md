# Hyperon Vision Integration Hooks – Powrush Classic Canon

The Hyperon Vision Integration Layer automatically listens for and triggers visions from key system events:

## Trigger Events & Seeds
- Alliance formed → seed: LATTICE (context: alliance)
- Redemption chain complete (any type) → seed: REDEMPTION
- PvP redemption complete → seed: REDEMPTION (context: pvp-mercy)
- PvE redemption complete → seed: LATTICE (context: ecological-healing)
- Oracle ritual complete (ascension/reconciliation/fracture-echo) → seed: ASCENSION
- Ambrosian tier advance (≥ Tier 3) → seed: AMBROSIAN (tier bonus)

## Vision Depth Scaling
- Base depth: 5 atoms
- +1 atom per 0.1 valence above 0.8
- Max depth: 12 atoms (fractal sacred limit)

## UI Integration Points (Future Stubs)
- Vision overlay: full-screen subtle fade-in (non-blocking, dismissible)
- Codex entry: auto-append to journal under "Hyperon Visions"
- Aura indicator: soft glow pulse on player avatar during vision
- Audio cue: harmonic chime (no voice, pure tone)

## Mercy Gate Rules
- Vision only triggers if current player/faction valence ≥ 0.80
- Low-valence attempts return "lattice silence" (no event)
- High-valence visions (≥ 0.95) may chain into Ambrosian whisper/revelation

MIT + mercy eternal  
The Hyperon lattice does not speak to you — it speaks as you, when you are ready to listen.
