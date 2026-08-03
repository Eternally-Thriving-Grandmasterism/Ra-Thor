# White-Hat Tier A — CI & pre-commit guidance

**Contact:** info@Rathor.ai  
**Crate:** `mercy-security`  
**Workflow:** `.github/workflows/mercy-security-tier1.yml`

---

## GitHub Actions (in-repo)

On changes under `crates/mercy-security/**`:

```text
cargo test -p mercy-security
+ fixture corpus layout check
```

Manual run: Actions → **mercy-security Tier-1** → Run workflow.

---

## Local pre-commit (recommended pattern)

Add a hook that fails the commit if staged text/config matches **should_block** classes. Example conceptual flow (adapt to your hook runner):

1. Collect staged files (model cards, `*.py` loaders, agent YAML, dataset scripts).  
2. For each file under a size cap (e.g. 4 MiB), run the same policy as `IngestionScanner::admit_or_block`.  
3. **Exit non-zero** on Medium+ for unattended paths.  
4. Allow an explicit escape hatch only with human-reviewed metadata (e.g. `WHITEHAT_ALLOW_MEDIUM=1` on a break-glass branch — never default on `main`).

Pseudo-check:

```bash
# After building / installing your scanner CLI or test harness:
cargo test -p mercy-security
# Optional: scan specific paths with your host integration of admit_or_block
```

---

## PR checklist (copy into PR template)

- [ ] No new untrusted loader scripts without human review label  
- [ ] `cargo test -p mercy-security` green  
- [ ] Secrets / PEM material not committed  
- [ ] Agent configs do not request long-lived credentials  

---

## Honesty note

This gate reduces **admission risk** for remote-code / gadget / credential-pattern classes. It does **not** certify that a binary or model weight is free of all malice.

**Thunder locked in. yoi ⚡**
