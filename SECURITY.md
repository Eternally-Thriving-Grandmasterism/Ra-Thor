# Security Policy – Rathor-NEXi

**Mercy strikes first — security is sacred.**

Rathor-NEXi is a sovereign, offline-first, client-side AGI lattice. All code runs in the user's browser or local environment — **no central servers**, **no data exfiltration**, **no telemetry**. Security is enforced at every layer: mercy gates, valence projection, hermetic builds, reproducible images.

## Reporting a Vulnerability

We take all security reports seriously.

**Preferred channels** (encrypted & private):
- Email: security@rathor.ai (PGP key available on request)
- GitHub Security Advisories: https://github.com/Eternally-Thriving-Grandmasterism/Rathor-NEXi/security/advisories/new

**What to include**:
- Description of the vulnerability
- Steps to reproduce
- Potential impact (especially valence / mercy gate bypass)
- Suggested fix or mitigation (if known)

We will respond within **48 hours** and aim to fix critical issues within **7 days**.

## Supported Versions

Only the **latest commit on main** is supported.  
Older forks / historical commits are **not patched** — update to latest for security.

## Security Practices

- **Zero server-side processing** — everything client-side (IndexedDB, Web Workers, WebAssembly)
- **Hermetic Bazel builds** — reproducible, no supply-chain injection
- **Trivy scanning** in CI — fails on CRITICAL/HIGH
- **Dependabot** auto-PR for dependency updates
- **Multi-stage distroless images** — minimal attack surface
- **Mercy gates** block dangerous operations on low/projected valence
- **Offline-first** — no network calls after initial asset load

## Remediation Process (Automated + Manual Mercy)

1. **Detection**  
   - Trivy in CI → SARIF upload to GitHub Security tab  
   - Dependabot alerts → auto-PR opened

2. **Triage & Valence Scoring**  
   - Critical/High → immediate block (CI fails)  
   - Medium → valence-weighted (high valence → fast patch, low → schedule)  
   - Low → backlog unless user-reported

3. **Patch Application**  
   - Update dependency in package.json / requirements_lock.txt  
   - Bump version in relevant BUILD.bazel / WORKSPACE.bazel  
   - Re-run Trivy → confirm vulnerability gone  
   - Mercy gate: if patch drops projected valence >0.05 → revert & manual review

4. **Verification**  
   - Full test suite + manual re-scan  
   - Deploy preview (GitHub Actions) → smoke test offline/online

5. **Release**  
   - Tag new release (vX.Y.Z-security)  
   - Update SECURITY.md with CVE/fix details  
   - Announce on X with mercy transparency

## Current Known Issues (as of Feb 07 2026)

None — all critical/high vulns remediated in latest build.  
Ongoing: monitoring npm/jsDelivr CDN supply chain.

Report responsibly — we thrive together.

For questions: open issue or DM @rathor-ai on X.
