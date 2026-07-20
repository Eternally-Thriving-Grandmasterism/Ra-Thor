# Ra-Thor Canonical Contact

**Primary contact for all licensing, security, partnerships, and inquiries:**

## info@Rathor.ai

### Deprecated (do not use)

| Deprecated address | Status |
|--------------------|--------|
| ceo@acitygames.com | Retired |
| CEO@ACITYGAMES.COM | Retired |
| AlphaProMega@ACityGames.com | Retired |
| INFO@ACITYGAMES.COM | Retired |
| info@ACityGames.com | Retired |

### Policy (as of 2026-07-20)

- All new files, Cargo.toml `authors`, LICENSE blocks, docs, and website footers use **info@Rathor.ai** only.
- Workspace default: `authors = ["Eternally-Thriving-Grandmasterism", "Sherif Samy Botros <info@Rathor.ai>"]`
- Brand / Grok–xAI posture: see [`docs/ATTRIBUTION_AND_BRAND.md`](docs/ATTRIBUTION_AND_BRAND.md) and [`docs/ONE_ORGANISM_GROK_FUSION.md`](docs/ONE_ORGANISM_GROK_FUSION.md). Ra-Thor does not claim xAI endorsement.

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
