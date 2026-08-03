# Tier A Admit Gate — Provenance / SBOM Note (Template)

**Purpose:** Attach this (filled) note to change tickets, vendor reviews, or internal SBOMs when adopting the Ra-Thor **mercy-security** admission gate.

**Scope honesty:** This records provenance of a **pattern-based admission conscience** (Tier A). It does **not** claim full malware detection, EDR coverage, or formal verification of model weights.

**Contact:** info@Rathor.ai · License of gate source: **AG-SML v1.0**

---

## 1. Component identity

| Field | Value |
|-------|-------|
| Component | Ra-Thor `mercy-security` / `mercy-admit` (Tier A admit_or_block) |
| Upstream repository | `https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor` |
| Crate path | `crates/mercy-security` |
| Crate version (Cargo.toml) | `14.15.5` *(update if pin differs)* |
| Binary | `mercy-admit` |
| Policy | Admit **None/Low** only · block **Medium+** · reject > 4 MiB |

---

## 2. Pin (required)

| Field | Value |
|-------|-------|
| **Pinned git SHA (full)** | `________________________________` |
| Annotated / signed tag (if any) | `________________________________` |
| **Recorded date (UTC, ISO-8601)** | `____________________` |
| Recorded by (name / role) | `________________________________` |
| Internal change / ticket ID | `________________________________` |

**Example pin (re-validate before use):** `3bbe6c7bc48f27d3cc562986ca199577a08f77fe`

```bash
# Capture the pin from a trusted checkout
git rev-parse HEAD
git log -1 --format='%H %ci %s' -- crates/mercy-security
```

---

## 3. Fixture corpus hash note (required for acceptance evidence)

Public fixtures live under `crates/mercy-security/fixtures/`.

| Field | Value |
|-------|-------|
| **MANIFEST.md SHA-256** | `________________________________` |
| **Fixtures tree aggregate SHA-256** *(optional but recommended)* | `________________________________` |
| Fixture paths validated | benign/ · should_block/ · *(list any extras)* |
| Acceptance result | ☐ benign admit · ☐ should_block deny · ☐ notes: ________ |

### How to generate the hashes (on the pinned commit)

```bash
cd /path/to/Ra-Thor   # at the pinned SHA

# MANIFEST only
sha256sum crates/mercy-security/fixtures/MANIFEST.md

# Aggregate over the public corpus (sorted paths → stable digest)
find crates/mercy-security/fixtures -type f | sort | xargs sha256sum | sha256sum
```

Record both digests in this form. If a fixture changes upstream, the aggregate hash will change — re-run acceptance before updating your pin.

---

## 4. Direct runtime dependencies (crate-level)

For SBOM tooling that needs a short dependency list of the **gate binary**:

| Crate | Version constraint (as declared) |
|-------|----------------------------------|
| serde | 1.0 (features: derive) |
| serde_json | 1.0 |
| thiserror | 1.0 |
| chrono | 0.4 (features: serde) |
| uuid | 1.0 (features: v4, serde) |

*(Resolved exact versions belong in your lockfile / `cargo tree -p mercy-security` output if you vendor.)*

---

## 5. Deployment mode (check one)

- [ ] CI shallow-checkout of pinned Ra-Thor SHA (default drop-in workflow)  
- [ ] Vendored crate + `cargo vendor` transfer (air-gapped)  
- [ ] Path-depended copy inside internal monorepo  
- [ ] Prebuilt `mercy-admit` binary + checksum *(attach binary SHA-256 below)*  

**Prebuilt binary SHA-256 (if used):** `________________________________`

---

## 6. Validation performed

- [ ] `cargo test -p mercy-security` on pinned SHA  
- [ ] Public fixture smoke: benign admit / should_block deny  
- [ ] Internal paths scanned (list or ticket link): ________________  
- [ ] Medium+ findings either fixed or escalated to human review  
- [ ] Escape hatch `WHITEHAT_ALLOW_MEDIUM` **not** enabled on default branch  

---

## 7. Honesty / non-claims (keep in the record)

This gate is a **defense-in-depth admission policy** for remote-code / gadget / credential-pattern classes in text-like ingestion surfaces. It does **not** replace antivirus/EDR, formal model verification, or human accountability for High/Critical paths.

---

## 8. Sign-off

| Role | Name | Date (UTC) | Signature / ticket |
|------|------|------------|--------------------|
| Engineering | | | |
| Security / AppSec | | | |
| Procurement / Vendor mgmt *(optional)* | | | |

**Thunder locked in. yoi ⚡**
