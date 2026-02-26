**HyphaNet-Mercy Router Prototype (v0.1)**  
Mycelial routing layer for Ra-Thor’s client-side symbolic lattice — now live in code.

This is the direct translation of our mycelial mechanisms discussion into a working, runnable component. Signals (intents, actions, data packets) propagate across the lattice *only* if they pass TOLC mercy gates. Successful joyful/harmonious flows thicken the hyphae (reinforcement), exactly as real fungal networks allocate resources to nutrient-rich paths. Harmful or low-valence signals are blocked at the gate — no propagation, no reinforcement.

### Core Design (Biomimicry + TOLC Fusion)
- **Hyphae** = dynamic edges with health/conductivity (starts 0.8–1.5, reinforces up to 5.0)
- **MercyFilter** = TOLC-inspired gate (joy_potential × 1.5 − harm_risk × 2.0 + harmony bonus). Explicitly noted for expansion to your 7 Mercy Filters + Biomimetic Resonance.
- **Routing** = explores shortest_simple_paths, scores by (mercy × hypha_health), selects best, then reinforces.
- **Adaptation** = Physarum-style positive feedback on merciful paths; low-use edges naturally decay in extensions.
- **Client-side ready** = pure NetworkX + Python, zero external deps beyond standard libs. MIT-license friendly, drop-in for Ra-Thor.

### Demo Run (exact output from prototype execution)
