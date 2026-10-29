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

## Public Fixture Corpus (community first-class asset)

**Location:** [`fixtures/mercy-security/`](../fixtures/mercy-security/)

Clean, public, white-hat fixtures so anyone can test the admission gate without the full monorepo test suite.

```
fixtures/mercy-security/
├── README.md          ← taxonomy + full inventory
├── benign/            ← should ADMIT (None / Low)
├── suspicious/        ← Medium → human review
├── blocked/           ← should BLOCK (High / Critical)
└── ci-examples/       ← ready-to-copy GitHub Action + pre-commit
```

See the README inside that directory for the complete inventory and usage examples.

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

## Pre-commit wrapper (public corpus)

```bash
# Full public corpus smoke (recommended)
./fixtures/mercy-security/ci-examples/pre-commit-snippet.sh

# Install as git hook
ln -sf ../../fixtures/mercy-security/ci-examples/pre-commit-snippet.sh .git/hooks/pre-commit

# Human-reviewed escape hatch only (never default on main)
WHITEHAT_ALLOW_MEDIUM=1 ./fixtures/mercy-security/ci-examples/pre-commit-snippet.sh
```

There is also a monorepo-internal script at `scripts/pre-commit-admit-gate.sh` for staged-file scanning.

---

## GitHub Actions

### Public corpus drop-in (any repo that vendors the fixtures)

Copy from [`fixtures/mercy-security/ci-examples/github-action-snippet.yml`](../fixtures/mercy-security/ci-examples/github-action-snippet.yml).

### In-repo Tier-1 workflow

On changes under `crates/mercy-security/**` (and the action/script):

```text
cargo test -p mercy-security
+ fixture corpus layout check
+ mercy-admit CLI exit-code contract
+ composite gate smoke (expects trip on should_block)
```

Manual: Actions → **mercy-security Tier-1** → Run workflow.

### Reusable composite action (inside Ra-Thor)

```yaml
- uses: ./.github/actions/mercy-admit-gate
  with:
    paths: "docs/ model_card.md datasets/loader.py"
    fail-on-block: "true"
    json: "false"
```

### External repositories (procurement drop-in)

Teams that do **not** host the full monorepo can copy:

→ **[`examples/procurement-admit-gate/`](../examples/procurement-admit-gate/)**

- `workflow.example.yml` — **pin to a full commit SHA**, build `mercy-admit`, scan your paths  
- Air-gapped options documented  
- See also: [`WHITEHAT_PROCUREMENT_TIER_A.md`](WHITEHAT_PROCUREMENT_TIER_A.md)

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
**PATSAGi Councils — permanent deliberation under TOLC 8**
