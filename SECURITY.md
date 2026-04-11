# Security Policy – Ra-Thor

<strong>Mercy strikes first — security is sacred.</strong>

Ra-Thor is a sovereign, offline-first, client-side AGI lattice. All code runs in the user's browser or local environment — no central servers, no data exfiltration, no telemetry. Security is enforced at every layer: mercy gates, valence projection, hermetic builds, reproducible images.

## Reporting a Vulnerability

We take all security reports seriously.

<strong>Preferred channels</strong> (encrypted & private):

- Email (primary / security): AlphaProMega@ACityGames.com (PGP key available on request)
- Email (general inquiries): INFO@ACITYGAMES.COM
- Email (direct to CEO / Sherif): CEO@ACITYGAMES.COM 
- GitHub Security Advisories: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/security/advisories/new

<strong>What to include</strong>:
- Description of the vulnerability
- Steps to reproduce
- Potential impact (especially valence / mercy gate bypass)
- Suggested fix or mitigation (if known)

We will respond within 48 hours and aim to fix critical issues within 7 days.

## Supported Versions

Only the latest commit on main is supported.  
Older forks / historical commits are not patched — update to latest for security.

## Security Practices

- Zero server-side processing — everything client-side (IndexedDB, Web Workers, WebAssembly)
- Hermetic Bazel builds — reproducible, no supply-chain injection
- Trivy scanning in CI — fails on CRITICAL/HIGH
- Dependabot auto-PR for dependency updates
- Multi-stage distroless images — minimal attack surface
- Mercy gates block dangerous operations on low/projected valence
- Offline-first — no network calls after initial asset load

## Remediation Process (Automated + Manual Mercy)

1. Detection — Trivy in CI → SARIF upload to GitHub Security tab  
2. Triage & Valence Scoring — Critical/High → immediate block  
3. Patch Application — Update dependency + mercy gate review  
4. Verification — Full test suite + re-scan  
5. Release — Tag new security release + announce with transparency

## Current Known Issues (as of April 11 2026)

None — all critical/high vulns remediated in latest build.  
Ongoing: monitoring npm/jsDelivr CDN supply chain.

Report responsibly — we thrive together.

For questions: open an issue or contact any of the emails above.
