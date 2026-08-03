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

3. (Recommended) Pin the Ra-Thor `ref:` to a known tag or commit once you have validated it.

4. Open a PR that touches a scanned path — the gate runs automatically.

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
