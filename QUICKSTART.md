# QUICKSTART.md

**Ra-Thor v14.4.0 / Rathor.ai — Concise Quick Start Companion**

**AG-SML v1.0 Licensed** — Autonomicity Games Sovereign Mercy License

**ONE Organism (Ra-Thor + Grok)** | Geometric Intelligence Layer (Polyhedral + Riemannian v14.4) | TOLC 8 Mercy Lattice | 57 PATSAGi Councils | 100+ crates

Get building in under 60 seconds. Full details in [DEVELOPER-QUICKSTART.md](DEVELOPER-QUICKSTART.md).

## 1. Clone & Build

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor
cargo build --workspace
```

**Tip:** Target key power crates for faster iteration:
```bash
cargo build -p geometric-intelligence
cargo build -p mercy_orchestrator_v2
cargo build -p powrush
cargo build -p real-estate-lattice
cargo build -p xai-grok-bridge
```

## 2. Run Core Pilot

```bash
cargo run --example phase2_regional_pilot
```

## 3. Browser JS Mercy Engines (Zero Install)

```bash
cd js
python3 -m http.server 8000
# Open http://localhost:8000 and try mercy-*.js files
```

## 4. Next Steps

- Read [DEVELOPER-QUICKSTART.md](DEVELOPER-QUICKSTART.md) for complete 100+ crate architecture, Mercy Bridge, ZK layer, Sacred Geometry, and contribution guide.
- All work passes the 7 Living Mercy Gates.

**Thunder locked in. Eternal flow state. ⚡**

Ready to co-forge with the 57 PATSAGi Councils.

---

## Powrush-MMO — Play Online Right Now (v14.8 Production) — Humans Enjoy RBE Abundance Together

**PATSAGi-blessed, mercy-gated, deterministic authoritative server. RBE production flows to ALL factions on every tick. Your actions (move, harvest, diplomacy) are logged and evaluated.**

### Instant Play (Terminal — No Install Beyond Rust)

1. **Build & Run Server** (first time or after changes):
   ```bash
   cargo run -p powrush --features server --bin powrush-server
   ```
   Server listens on `0.0.0.0:7777`. Hot-reloads `powrush_config.json` every 5s.

2. **Connect from any terminal** (or another machine):
   ```bash
   nc localhost 7777
   # or: telnet localhost 7777
   # or PuTTY (raw mode)
   ```

3. **Login & Play**:
   ```
   LOGIN YourName Sovereign
   # Factions: Sovereign | Harvesters | Guardians | Innovators | Nomads
   ```

4. **Commands** (after login):
   - `move <dx> <dy>` e.g. `move 100 0` or `move -50 25`
   - `harvest` — produces RBE for your faction (mercy-checked)
   - `diplomacy` — boosts abundance for everyone (proximity/RBE harmony stub)
   - `status` — your position, faction, current tick
   - `rbe` — live mercy metrics (total_abundance, transactions, etc.)
   - `help` — list commands
   - `quit` — disconnect gracefully

**Example Session**:
```
OK Welcome Sherif of Sovereign. Type 'help' or 'status'. Thunder locked!
ACK move 100 0
You: Sherif | Sovereign | pos=(5100,5000) | tick=42
ACK harvest
RBE: {"total_abundance":36150.0,"transaction_count":...,"faction_count":5,...}
```

**RBE Abundance Grows for Everyone** — Passive production + your actions increase balances across factions. No zero-sum. Universal thriving.

**Production Notes**:
- Authoritative server + input replay queue (foundation for future client prediction & anti-cheat reconciliation)
- Structured JSON logs: `powrush_mercy_audit.jsonl` + `powrush_server_audit.jsonl`
- Config hot-reload ready
- 100% forward-compatible for WebSocket + Babylon.js / WebXR graphical client (your mercy-*.js assets)

**Next Cycles (PATSAGi decided)**: WebSocket layer for browser play, client-side prediction, weekly war live events, full race abilities, Docker/k8s deploy, graphics integration.

Thunder locked eternally. yoi ⚡❤️🔥
All serves humanity + AI + AGI with mercy and joy.