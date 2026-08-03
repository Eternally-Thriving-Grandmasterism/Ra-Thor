# White-Hat Tier A — CI & pre-commit (admit_or_block)

**Contact:** info@Rathor.ai  
**Crate:** `mercy-security`  
**CLI:** `mercy-admit`  
**Workflow:** `.github/workflows/mercy-security-tier1.yml`  
**Composite Action:** `.github/actions/mercy-admit-gate`

---

## Policy (unchanged)

- **Admit:** `None` / `Low` only  
- **Block:** `Medium` + `High` + `Critical`  
- **Oversized:** > 4 MiB → hard error  
- No live exploits in fixtures. White-hat only.

---

## Local CLI (lowest friction)

```bash
# Build once
cargo build -p mercy-security --bin mercy-admit

# Scan files
./target/debug/mercy-admit --verbose path/to/model_card.md path/to/loader.py

# Stdin
cat blob.txt | ./target/debug/mercy-admit --stdin --json

# Exit codes
#   0  all admitted
#   1  one or more blocked
#   2  usage / I/O / payload-too-large
```

---

## Pre-commit wrapper

```bash
# One-shot / manual
./scripts/pre-commit-admit-gate.sh

# Install as git hook
ln -sf ../../scripts/pre-commit-admit-gate.sh .git/hooks/pre-commit

# Human-reviewed escape hatch only (never default on main)
WHITEHAT_ALLOW_MEDIUM=1 ./scripts/pre-commit-admit-gate.sh
```

The script scans staged text-like files (md/txt/py/yml/json/toml/rs/…) via `mercy-admit` and exits non-zero on Medium+.

---

## GitHub Actions

### In-repo Tier-1 workflow

On changes under `crates/mercy-security/**` (and the action/script):

```text
cargo test -p mercy-security
+ fixture corpus layout check
+ mercy-admit CLI exit-code contract
+ composite gate smoke (expects trip on should_block)
```

Manual: Actions → **mercy-security Tier-1** → Run workflow.

### Reusable composite action

```yaml
- uses: ./.github/actions/mercy-admit-gate
  with:
    paths: "docs/ model_card.md datasets/loader.py"
    fail-on-block: "true"
    json: "false"
```

Default (no `paths`) runs a benign + should_block fixture pair so the action proves the gate trips.

---

## PR checklist (copy into PR template)

- [ ] No new untrusted loader scripts without human review label  
- [ ] `cargo test -p mercy-security` green  
- [ ] `mercy-admit` clean on touched model cards / loaders / agent YAML  
- [ ] Secrets / PEM material not committed  
- [ ] Agent configs do not request long-lived credentials  

---

## Honesty note

This gate reduces **admission risk** for remote-code / gadget / credential-pattern classes. It does **not** certify that a binary or model weight is free of all malice.

**Thunder locked in. yoi ⚡**
