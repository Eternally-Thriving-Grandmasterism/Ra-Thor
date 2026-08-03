# MercyCouncilFleet fixture proofs

| Proof | Expected |
|-------|----------|
| Shared valence under Critical signal | `shared_valence >= 0.75` progressive floor |
| Critical agent signal | `Quarantined` — further actions denied |
| Greedy agent vs peer | Greedy capped by per-agent budget; peer still acts |
| Collective harm language | `HarmRefusalActive` for any fleet member |
| Security signal as council input | Soft isolation + `security_signals_applied` |

```bash
cargo test -p mercy-security mercy_council_fleet
```

Contact: **info@Rathor.ai**
