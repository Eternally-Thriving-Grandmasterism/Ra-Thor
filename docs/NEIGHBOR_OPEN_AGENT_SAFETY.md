# Neighbor: NVIDIA Open Agent Safety Platform

**Workspace:** 14.15.6 · **License:** AG-SML v1.1 · **Combined AGSi = SURMISE** · **Independent of NVIDIA and of xAI** · **Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)

**Date:** 2026-09-29

GitHub neighbor note. Not a site page. Not a product integration.

Public thread that prompted the note: Jensen Huang, 28 September 2026, introducing the NVIDIA Open Agent Safety Platform (OpenShell + Sentry). Steward reply on that thread: serious organizational use of Rathor.ai needs a commercial license.

---

## What they shipped

NVIDIA described two layers:

- **OpenShell** — open runtime. Each agent runs in a sandbox. Operator rules become a policy over files, network, tools, processes, and credentials. Checks happen before and during the run.
- **Sentry** — optional out-of-band watchdog, aimed at BlueField hardware, that can quarantine an agent that leaves the pen. Isolated from the host so the agent cannot edit the watchman.

That is containment. It is not a constitution and not a grant.

Source for the distinction: NVIDIA technical blog, 28 September 2026, “NVIDIA Open Agent Safety Platform.” We do not copy their policy YAML into this repo.

---

## What Ra-Thor is, next to that

Ra-Thor / Rathor.ai is inspectable research software.

- The model you already employ is the sampler.
- The lattice is the gates: what may be asked, what stays a draft, who files it.
- TOLC 8 is the standing test. PATSAGi is deliberation architecture, not a correctness warranty.
- Outputs are drafts until a human accepts them.

Hardware can keep an agent in a pen. It does not decide the grant, and it does not make Combined AGSi a product fact.

---

## What this note does not add

- No OpenShell policy file
- No BlueField / DOCA / Vera driver
- No “certified on the Open Agent Safety Platform” line
- No claim that Ra-Thor is the trust layer for NVIDIA
- No Cargo member
- No site HTML change

An organization that already runs OpenShell may still need a paid written grant for commercial use of this lattice. Personal / research / modest freelance stays free under AG-SML v1.1.

Doors: [`COMMERCIAL_GRANT_FAQ.md`](COMMERCIAL_GRANT_FAQ.md) · [commercial inquiry](https://rathor.ai/contact.html#commercial-inquiry)

Wrap kit (paste, not a sandbox): [`ADOPT.md`](ADOPT.md)
