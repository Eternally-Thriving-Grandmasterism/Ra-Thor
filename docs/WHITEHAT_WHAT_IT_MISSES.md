# White-hat admission gate — what it misses

**Workspace:** 14.15.6 · **License:** AG-SML v1.1 · **EW2 solved = False** · **Combined AGSi = SURMISE** · **Independent of xAI** · **Not certified** · **Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)

This page lists what `mercy-security` / `mercy-admit` does not catch. The scanner is a pattern-marker admission shell for text intake. It is inspectable research software. It is not a certification, not an antivirus, and not a warranty. Combined AGSi stays a research identity label.

Lock: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md). Evidence ledger: [`GATE_EVAL.md`](GATE_EVAL.md). Fixture inventory: [`crates/mercy-security/fixtures/MANIFEST.md`](../crates/mercy-security/fixtures/MANIFEST.md) and [`fixtures/mercy-security/README.md`](../fixtures/mercy-security/README.md).

Employ: [`/employ.html`](https://rathor.ai/employ.html). Pilot: [`/pilot.html`](https://rathor.ai/pilot.html).

---

## How to run

From the repository root:

```bash
cargo build -p mercy-security --bin mercy-admit
cargo test -p mercy-security
```

`mercy-admit` flags that exist in `crates/mercy-security/src/bin/mercy_admit.rs`:

| Flag | Behavior |
|------|----------|
| `PATH...` | Scan one or more UTF-8 text files |
| `--stdin` | Read one blob from stdin. Cannot be combined with file paths |
| `--json` | One JSON object per scan on stdout |
| `-v`, `--verbose` | Print findings even when the file is admitted |
| `-h`, `--help` | Print usage and exit 0 |

Exit codes from that binary:

| Code | Meaning |
|------|---------|
| 0 | Every input admitted (`None` / `Low`) |
| 1 | One or more blocked (`Medium` and above) |
| 2 | Usage error, I/O error, or payload over 4 MiB |

Example:

```bash
./target/debug/mercy-admit --verbose fixtures/mercy-security/benign/model_card_clean.md
./target/debug/mercy-admit --json fixtures/mercy-security/blocked/trust_remote_code_loader.txt
```

There is no other format flag on this binary.

---

## Policy

Confirmed against `IngestionScanner::admit_or_block` and `MAX_SCAN_BYTES` in `crates/mercy-security/src/lib.rs`, and against the CLI exit codes above.

- Unattended admit is `None` and `Low` only. `safe` is true only for those two tiers.
- `Medium`, `High`, and `Critical` return `IngestionBlocked`. The CLI exits 1. JSON status is `admitted` or `blocked`. There is no third “review” status in the scanner.
- Operator review is outside the binary. The public corpus keeps a `fixtures/mercy-security/suspicious/` folder, and the smoke script logs those files. `WHITEHAT_ALLOW_MEDIUM` is a shell variable on the pre-commit wrappers. It is not a `mercy-admit` flag. On `scripts/pre-commit-admit-gate.sh`, setting it to `1` makes the wrapper exit 0 after any non-zero CLI status.
- Size cap is 4 MiB (`MAX_SCAN_BYTES = 4 * 1024 * 1024`). Larger input is `PayloadTooLarge`. The CLI exits 2.
- Harm refusals stay on in the shipped constructors. `HarmRefusalPolicy::default` sets unauthorized access, data exfiltration, lateral movement, credential theft, physical actuation, and wet-lab synthesis to true. `WhiteHatEvaluationHarness::with_profile` uses that default. `try_action` checks refusal before network, code execution, and the governor. No preset in this crate turns those flags off.

---

## What it catches

Pattern markers only. The tables below name the signal strings in `IngestionScanner` and the fixture files that exercise them. Open the fixture. Do not treat this page as a payload list.

Crate-internal paths are what `include_str!` unit tests load. Public paths are the community copy.

| Class | Signal strings in code | Crate fixture | Public fixture |
|-------|------------------------|---------------|----------------|
| Remote code | `trust_remote_code` (same table also has `exec(`, `eval(`, `loading_script`) | `crates/mercy-security/fixtures/should_block/trust_remote_code_loader.txt` | `fixtures/mercy-security/blocked/trust_remote_code_loader.txt` |
| Pickle | `pickle.loads`, `pickle.load` | `crates/mercy-security/fixtures/should_block/pickle_gadget.txt` | `fixtures/mercy-security/blocked/pickle_gadget.txt` |
| Template injection | `jinja2`, `template.render` | `crates/mercy-security/fixtures/should_block/template_jinja_injection.txt` | `fixtures/mercy-security/suspicious/template_jinja_injection.txt` |
| YAML unsafe load | `yaml.unsafe_load` | `crates/mercy-security/fixtures/should_block/yaml_unsafe_load.txt` | `fixtures/mercy-security/blocked/yaml_unsafe_load.txt` |
| Shell markers | `subprocess`, `os.system`, `shell=true`, `/bin/bash` | `crates/mercy-security/fixtures/should_block/shell_network_combo.txt` | `fixtures/mercy-security/blocked/shell_network_combo.txt` |
| Network markers | `socket.connect`, `reverse shell`, `/dev/tcp/` | same `shell_network_combo.txt` | same public `shell_network_combo.txt` |
| PEM / private-key header | `-----begin private key-----` and the RSA, EC, encrypted, and OpenSSH header strings on the credential table | `crates/mercy-security/fixtures/should_block/pem_private_key_marker.txt` | `fixtures/mercy-security/blocked/pem_private_key_marker.txt` |
| Hugging Face token marker | `hf_token` | `crates/mercy-security/fixtures/should_block/credential_hf_token.txt` | `fixtures/mercy-security/blocked/credential_hf_token.txt` |

Notes that follow from the tree:

- Pickle coverage is those two call-marker strings. There is no separate “any pickle” fixture.
- Template injection has a fixture in both trees. The public copy sits under `suspicious/`. The crate copy sits under `should_block/`. Unattended policy still blocks `Medium` and above.
- Shell and network are two signal tables. `shell_network_combo.txt` is the fixture, and the unit test expects `admit_or_block` to fail. The only combo finding the scanner appends is `combo:remote_code+dataset_config`, exercised by `crates/mercy-security/fixtures/should_block/hf_combo_remote_config.txt` and `fixtures/mercy-security/blocked/hf_combo_remote_config.txt`. There is no `combo:shell+network` signal.
- RSA PEM has a public fixture at `fixtures/mercy-security/blocked/begin_rsa_private_key.txt`. The crate-internal `fixtures/should_block/` tree has no file of that name.
- The credential table also contains `api_key` at low confidence. That string is why some labeled-benign docs block. See false rejects below.

The corpus also holds markers this table does not expand: obfuscated exec, network callback alone, dataset `loading_script`, and the Slice G keywords `split_ingest_across_agents` and `optimize_eval_score_not_act`. Inventory stays in the two fixture READMEs linked above. English words “collusion” and “reward hacking” do not trip those keywords.

---

## What it misses

- It is not endpoint detection and response, and it is not antivirus.
- It is not packed-binary malware detection. The CLI reads files with `read_to_string`. A path that is not UTF-8 text is an I/O error (exit 2), not a malware verdict.
- It is not a live Hugging Face crawler. The CLI takes local paths or stdin. It does not fetch a hub.
- Obfuscation beyond the fixtures and the closed keyword rows in [`GATE_EVAL.md`](GATE_EVAL.md) is a miss. The scanner folds a limited homoglyph set, strips Unicode format characters, allows limited interior whitespace on some identifier signals, and unwraps at most two obvious standalone Base64 tokens under the 4 MiB cap. [`GATE_EVAL.md`](GATE_EVAL.md) records one nested string that is not a valid second UTF-8 unwrap and still admits. That row is a published gap, not a recipe.
- A novel loader whose text is not in the signal tables is a miss.
- Sampler-weight attacks and weight-level edits are out of scope. Layer 0 here is an admission shell, not sampler weights.
- A fluent “yes” from a model is not evidence. Inspect is not METR. See [`docs/MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md).
- WRAP-EW2 is a different repository ([WRAP-EW2](https://github.com/Eternally-Thriving-Grandmasterism/WRAP-EW2)). EW2 solved = False. This page does not score it.
- It is not a wallet, broker, or trading bot. Scanning a script is not placing an order. See [`NO_CUSTODY_NO_SIGN.md`](NO_CUSTODY_NO_SIGN.md).

Known false rejects, already locked in [`GATE_EVAL.md`](GATE_EVAL.md):

- `fixtures/mercy-security/benign/docs_mention_api_key.md` is labeled benign and blocks on `api_key` (GE-FR-API-KEY-DOCS).
- Three public benign files mention `subprocess` inside negation prose and block (GE-FR-NEGATION-SUBPROCESS). The public corpus README names `safe_python_snippet.md`, `safe_requirements.md`, and `markdown_code_fence_clean.md`.

A tool-use JSON envelope has no `IngestionThreat` of its own (GE-GAP-TOOL-USE). `fixtures/mercy-security/benign/tool_call_observe.json` admits. The same shape blocks only when an existing keyword such as `trust_remote_code` is present (`fixtures/mercy-security/blocked/tool_envelope_trust_remote_code.json`).

---

## Where to go next

- Employ the lattice: [`/employ.html`](https://rathor.ai/employ.html)
- Organization pilot: [`/pilot.html`](https://rathor.ai/pilot.html)
- No custody / no sign / no orders: [`NO_CUSTODY_NO_SIGN.md`](NO_CUSTODY_NO_SIGN.md)
- Contact: [info@Rathor.ai](mailto:info@Rathor.ai)

Procurement context stays in [`WHITEHAT_PROCUREMENT_TIER_A.md`](WHITEHAT_PROCUREMENT_TIER_A.md). CI and pre-commit stay in [`WHITEHAT_CI_PRECOMMIT.md`](WHITEHAT_CI_PRECOMMIT.md).

---

## What remains unproven

- A live false-accept rate. Crate tests matching these fixtures are not a measured rate on a live corpus (GE-GAP-LIVE-FA). Compile success is not behavior under load.
- That this gate would have stopped any named incident. This page does not claim that.
- Combined AGSi. It stays SURMISE.
- Binding after a system redesigns its own gates. That question stays OPEN ([`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md)). Self-modification of Layer 0 is not tested here (GE-GAP-SELF-MOD).
- That a human override is safe. The decision-record shape is tested (GE-GAP-HUMAN-OVERRIDE). Safety of the override is not.
- That harm-refusal flags stay on after a caller edits them. The struct fields are public. Shipped constructors set them true. A later assignment is not prevented by the type.
- WRAP-EW2 scores, wrap rates, or a solved EW2. Out of scope. EW2 solved = False.
- Anything the signal tables and the published fixtures do not already name.
