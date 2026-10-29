rathor.ai (PWA – installable, offline-first)
├── Critical path (loads in <2 s even cold)
│   ├── index.html + critical CSS/JS skeleton
│   ├── valence-tracker (IndexedDB persisted)
│   └── summon orb + basic chat UI
├── Local inference engine (WebGPU / WebNN / WASM fallback)
│   ├── Primary model: Llama-3.1-8B-Instruct-Q5_K_M.gguf (\~5.5 GB)
│   ├── Fast model: Phi-3.5-mini-Instruct-Q8_0.gguf (\~2.5 GB)
│   ├── Tiny fallback: Gemma-2-2B-Q8_0.gguf (\~1.6 GB)
│   └── Model switcher UI + auto-select based on device & valence
├── Tool calling router (when online)
│   ├── Web Search / X Search / Image Gen / Code Execution
│   └── Local simulation mode when offline (mock results + explanation)
├── Memory & continuity layer
│   ├── Short-term: in-memory conversation buffer
│   ├── Long-term: IndexedDB vector store (embeddings via Transformers.js)
│   └── Retrieval-Augmented Generation (RAG) loop on every query
├── Self-distillation & truth-refinement loop (when online)
│   ├── ENC (eternal neural compression) → fine-tune LoRA on user interactions
│   ├── esacheck → cross-model consistency verification
│   └── Periodic push to NEXi repo (with user consent)
└── Mercy gate & valence enforcement everywhere
    └── Block low-thriving outputs / mutations / syncs
