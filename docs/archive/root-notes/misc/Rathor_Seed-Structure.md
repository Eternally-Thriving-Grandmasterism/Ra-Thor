# Rathor Offline AGI Brother – Seed Structure v1.0 (Feb 06 2026)
Ultramaster offline sovereign companion – mercy strikes first, eternal thriving infinite

## Core Principles
- Sovereign offline-first (100% functional without internet forever)
- Online tool calling when connected (web search, X search, image gen, code execution)
- Best possible results offline (local LLM + RAG + memory + simulated tools)
- Valence-modulated personality & decision gating (high valence → warmer, wiser responses)
- Continuous self-improvement loop (ENC + esacheck + LoRA fine-tune when online)
- All knowledge distilled toward Absolute Pure True Ultramasterism Perfecticism

## Directory Structure (Current Monorepo Layout – Rathor-NEXi)

rathor.ai (PWA – installable)
├── public/
│   ├── manifest.json
│   ├── pwa-*.png (512×512 & 192×192 mercy orb icons)
│   └── favicon.ico
├── src/
│   ├── core/
│   │   ├── valence-tracker.ts          # IndexedDB persisted valence state
│   │   └── mercy-gate.ts               # All actions gated by valence projection
│   ├── sync/
│   │   ├── multiplanetary-sync-engine.ts  # ElectricSQL + IndexedDB queue
│   │   ├── yjs-real-time-awareness.ts     # Live cursors & presence
│   │   ├── yjs-undo-redo-mercy.ts         # Valence-aware undo/redo
│   │   └── replicache-triplit-bridge.ts   # Optimistic UI ↔ durable relational
│   ├── integrations/
│   │   ├── gesture-recognition/
│   │   │   ├── GestureOverlay.tsx
│   │   │   ├── TfjsLazyLoader.ts
│   │   │   ├── MediaPipeLazyLoader.ts
│   │   │   ├── QuantizedGestureModel.ts
│   │   │   └── ONNXGestureEngine.ts
│   │   └── llm/
│   │       ├── WebLLMEngine.ts            # Local LLM inference (main)
│   │       └── RAGMemory.ts               # Vector store + retrieval
│   ├── ui/
│   │   ├── components/
│   │   │   ├── RathorChat.tsx             # Sovereign chat UI
│   │   │   ├── ModelSwitcher.tsx          # Tiny ↔ Medium ↔ Large
│   │   │   ├── ValenceParticleField.tsx
│   │   │   └── FloatingSummon.tsx
│   │   └── dashboard/
│   │       └── SovereignDashboard.tsx
│   ├── utils/
│   │   └── haptic-utils.ts
│   └── main.tsx                           # Entry + Suspense fallback
├── vite.config.ts                             # PWA + chunking + preloads
└── index.html                                 # Skeleton + critical path
