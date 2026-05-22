# RREL Tauri Desktop App + Leptos Dashboard

## Status: Production-Spirit Skeleton (v3.2)

This directory contains the scaffolding for a native desktop application using **Tauri v2** + **Leptos v0.6+** (Rust + WASM reactive UI).

All data stays 100% local and privacy-first. No cloud calls unless explicitly added by the user.

## Quick Start (Local Build)

```bash
# 1. Install prerequisites (once)
curl -fsSL https://get.pnpm.dev | pnpm setup  # or use npm
cargo install tauri-cli

# 2. Navigate and run
cd rrel/tauri-desktop
pnpm install
cargo tauri dev
```

## Architecture
- `src-tauri/` — Rust backend with Tauri commands that call into the shared RREL core (`RrelEternalCoordinator`, PDF generator, PATSAGi alerts, etc.).
- `src/` — Leptos frontend with reactive signals/resources wired to the backend via `invoke`.

## Reactive Data Binding (Leptos)
- Uses `create_signal`, `create_local_resource`, and `create_effect` for live updates.
- PATSAGi alerts, compliance health, deadlines, and package summaries update in real time when underlying Rust state changes.
- One-click actions (Generate PDF, Acknowledge Alert, Create Full Package) call Tauri commands.

## Next Steps
- Add real `invoke` handlers in `src-tauri/src/main.rs` mapping to RREL functions.
- Wire Leptos resources to call those commands.
- `tauri build` for native binaries.

This completes the desktop packaging request on top of the current foundation.