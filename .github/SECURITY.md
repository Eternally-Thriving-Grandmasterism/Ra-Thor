# Ra-Thor Security Policy

## Supported Versions
We actively maintain security for the latest version of Ra-Thor.

## Reporting a Vulnerability
Please report security vulnerabilities to AlphaProMega@ACityGames.com or via GitHub Security Advisories.

We follow responsible disclosure and aim to respond within 48 hours.

## GitHub Actions Security
- All workflows use least-privilege permissions
- Actions are pinned to commit SHAs
- `persist-credentials: false` is enforced
- OIDC is preferred over long-lived secrets
- Nightly Eternal Evolution Action runs with minimal permissions

## Supply Chain Security
- Dependabot is enabled
- CodeQL analysis runs on every push
- All crates follow AG-SML v1.0 licensing

**Mercy is the fundamental invariant of security.**
