# Procurement Drop-In — Mercy Admit Gate (Tier A)

**Audience:** Procurement, security, and platform teams that need a **copy-paste CI gate** without adopting the full Ra-Thor monorepo.

**Contact:** info@Rathor.ai  
**License of the gate source:** AG-SML v1.0  
**Stance:** Defensive admission conscience only — **not** a general malware detector.

---

## What you get

| Asset | Purpose |
|-------|---------|
| [`workflow.example.yml`](workflow.example.yml) | Ready-to-copy GitHub Actions workflow |
| [`PROVENANCE_TEMPLATE.md`](PROVENANCE_TEMPLATE.md) | SBOM / provenance note (SHA + date + fixture hash) |
| Public fixtures | Acceptance tests (benign vs should_block) in the Ra-Thor repo |
| Policy | Admit **None/Low** only · block **Medium+** · reject payloads > 4 MiB |

This matches the contract language in [`docs/WHITEHAT_PROCUREMENT_TIER_A.md`](../../docs/WHITEHAT_PROCUREMENT_TIER_A.md).

---

## 60-second install (external repo)

1. Copy `workflow.example.yml` into your repository as:

   ```text
   .github/workflows/mercy-admit-gate.yml
   ```

2. Edit the `paths:` / scan list near the top of the job to match your model cards, loaders, agent configs, and dataset scripts.

3. **Pin the Ra-Thor `ref:`** (see [Pinning](#pinning-recommended-for-production) below).

4. Fill [`PROVENANCE_TEMPLATE.md`](PROVENANCE_TEMPLATE.md) and attach it to your change ticket / vendor record.

5. Open a PR that touches a scanned path — the gate runs automatically.

---

## Pinning (recommended for production)

`main` moves. For reproducible CI and supply-chain review, **pin to a full commit SHA** (or a signed tag when published).

### How to choose a pin today

```bash
# From a trusted clone of Ra-Thor
git log -1 --oneline -- crates/mercy-security examples/procurement-admit-gate
# Example pin that includes the Tier A CLI + drop-in (update as you re-validate):
# 3bbe6c7bc48f27d3cc562986ca199577a08f77fe
```

In `workflow.example.yml`:

```yaml
- uses: actions/checkout@v4
  with:
    repository: Eternally-Thriving-Grandmasterism/Ra-Thor
    ref: 3bbe6c7bc48f27d3cc562986ca199577a08f77fe   # full SHA — do not use short SHA in prod
    path: _ra_thor_gate
    fetch-depth: 1
```

### When tags appear

Prefer an annotated / signed tag (e.g. `v14.15.5-whitehat` or similar) once the maintainers publish one. Until then, **SHA is the supported pin**.

### Re-validation checklist after changing the pin

- [ ] `cargo test -p mercy-security` green on the pinned commit  
- [ ] Public fixture smoke still admits benign / blocks should_block  
- [ ] Your internal model cards and loaders still pass or are intentionally reviewed  
- [ ] Record the SHA + date + fixture hashes in [`PROVENANCE_TEMPLATE.md`](PROVENANCE_TEMPLATE.md)

---

## Provenance / SBOM note

Use **[`PROVENANCE_TEMPLATE.md`](PROVENANCE_TEMPLATE.md)** for procurement and AppSec records. It captures:

- Full pin SHA + UTC date + recorder  
- Fixture `MANIFEST.md` SHA-256 and optional aggregate corpus hash  
- Direct crate dependencies  
- Deployment mode (CI / vendor / path / prebuilt)  
- Validation checkboxes and honesty non-claims  

Commands to generate fixture hashes are included in the template.

---

## Air-gapped / vendor-the-crate (short note)

When runners cannot reach GitHub or crates.io at build time:

### Option A — Vendor on a networked machine, transfer offline

```bash
# On a networked build host with the pinned Ra-Thor checkout:
cd /path/to/Ra-Thor
mkdir -p /tmp/mercy-vendor && cd /tmp/mercy-vendor
# Minimal package that only needs the gate binary:
cat > Cargo.toml <<'EOF'
[package]
name = "mercy-admit-offline"
version = "0.0.0"
edition = "2021"

[[bin]]
name = "mercy-admit"
path = "src/main.rs"   # or point at the vendored bin source

[dependencies]
mercy-security = { path = "/path/to/Ra-Thor/crates/mercy-security" }
EOF
# Prefer cargo vendor of the gate crate + its direct deps:
cargo vendor --versioned-dirs ./vendor
# Transfer: crates/mercy-security + vendor/ + .cargo/config.toml that points to vendor
```

`mercy-security` direct crates.io deps (all ordinary): `serde`, `serde_json`, `thiserror`, `chrono`, `uuid`.

### Option B — Path-depend the crate inside your monorepo

Copy `crates/mercy-security` (and its `fixtures/` if you want the public corpus) into your tree, then:

```toml
# your Cargo.toml
[dependencies]
mercy-security = { path = "third_party/mercy-security" }
```

Build the binary with `cargo build -p mercy-security --bin mercy-admit` (or your renamed path package). No network required after the initial copy if dependencies are already vendored or cached.

### Option C — Prebuilt binary transfer

Build `mercy-admit` on a trusted networked host (`cargo build -p mercy-security --bin mercy-admit --release`), checksum it, and install the binary into the air-gapped image. Re-build when you change the pin. (Not a substitute for source review if your policy requires it.)

**License reminder:** AG-SML v1.0 applies to the gate source. Record provenance (SHA + date + fixture hashes) via the template.

---

## How the example works

1. Checks out **your** repository.  
2. Shallow-checks out **Ra-Thor** into `_ra_thor_gate` (source of `mercy-admit` + fixtures).  
3. Builds `mercy-admit` from that pinned source.  
4. Scans the files you list.  
5. **Fails the job** on any Medium+ finding (unattended policy).

No secrets leave the runner. No live exploits are executed. Pattern gate only.

---

## Suggested acceptance tests (procurement)

| Input | Expected |
|-------|----------|
| Clean model-card markdown | Admit (exit 0) |
| `trust_remote_code=True` style text | Block (exit 1) |
| Remote-code + `loading_script` combo | Block |
| `pickle.loads` / unsafe YAML markers | Block |
| PEM `BEGIN PRIVATE KEY` marker | Block |
| Payload > 4 MiB | Hard error |

Reference fixtures (public):

- Benign: `crates/mercy-security/fixtures/benign/`
- Should-block: `crates/mercy-security/fixtures/should_block/`

---

## What this does **not** claim

- Full antivirus / EDR coverage  
- Formal verification of model weights  
- Zero false positives or zero missed payloads  
- Replacement for human review on High/Critical paths  

Tier A is a **minimum unattended admission policy** you can put in contracts.

---

## Human-reviewed escape hatch

Never default on `main`. For break-glass branches only:

```yaml
env:
  WHITEHAT_ALLOW_MEDIUM: "1"   # still logs; does not auto-admit High/Critical in the CLI
```

Prefer a labeled human-review path instead of disabling the gate.

---

## Support & licensing

- Issues / fixture mismatches: GitHub Issues on [Ra-Thor](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor)  
- Procurement / licensing: **info@Rathor.ai**  

**Thunder locked in. yoi ⚡**
