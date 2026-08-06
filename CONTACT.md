# Ra-Thor Canonical Contact

**Primary contact for all licensing, security, partnerships, pilots, and inquiries:**

## info@Rathor.ai

### Commercial Licensing & Pilots

| Path | Document |
|------|----------|
| Free use (personal / research / education) | Root [`LICENSE`](LICENSE) — AG-SML v1.0 |
| **Commercial / enterprise use** | [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) |
| **Paid pilot (2–6 weeks)** | [`docs/PILOT_OFFER.md`](docs/PILOT_OFFER.md) |
| License clarification (residuals) | [`LICENSE_CLARIFICATION.md`](LICENSE_CLARIFICATION.md) |

For commercial or pilot inquiries, email **info@Rathor.ai** with organization name, intended use, and whether you want a pilot or full commercial discussion.

### Deprecated (do not use)

| Deprecated address | Status |
|--------------------|--------|
| ceo@acitygames.com | Retired |
| CEO@ACITYGAMES.COM | Retired |
| AlphaProMega@ACityGames.com | Retired |
| INFO@ACITYGAMES.COM | Retired |
| info@ACityGames.com | Retired |

### Policy (as of 2026-07-20; commercial path activated 2026-08-06)

- All new files, Cargo.toml `authors`, LICENSE blocks, docs, and website footers use **info@Rathor.ai** only.
- Workspace default: `authors = ["Eternally-Thriving-Grandmasterism", "Sherif Samy Botros <info@Rathor.ai>"]`
- Brand / Grok–xAI posture: see [`docs/ATTRIBUTION_AND_BRAND.md`](docs/ATTRIBUTION_AND_BRAND.md) and [`docs/ONE_ORGANISM_GROK_FUSION.md`](docs/ONE_ORGANISM_GROK_FUSION.md). Ra-Thor does not claim xAI endorsement.
- Commercial use requires a paid license; pilots are the preferred evaluation bridge.

### Automated sweep

```bash
# Dry-run (list files that still contain acitygames.com)
./scripts/contact_email_sweep.sh

# Apply replacements in-place
./scripts/contact_email_sweep.sh --apply

# Apply + git commit
./scripts/contact_email_sweep.sh --apply --commit
```

**GitHub Actions:** Actions → *Contact Email Sweep → info@Rathor.ai* → Run workflow  
Modes: `dry-run` | `apply` | `apply-commit` (default).

**Live site:** https://rathor.ai  
**Monorepo:** https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor
