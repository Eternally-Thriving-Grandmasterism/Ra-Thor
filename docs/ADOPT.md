# Adopt / wrap a model

**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Spine:** [`EMPLOY.md`](EMPLOY.md) · [`OPTIONAL_MODEL.md`](OPTIONAL_MODEL.md)  
**Lock:** [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md)

How a stranger wraps Grok, Claude, ChatGPT, Gemini, Ollama, or any OpenAI-compatible host under Ra-Thor. Outputs remain **drafts**. inspect ≠ METR. Independent of xAI.

## Four doors (pick one)

| Door | Who | What |
|------|-----|------|
| 1. Copy context | Anyone | Lattice Chat **Copy Context**, or paste [`wrappers/system-prompt.txt`](../wrappers/system-prompt.txt) into the model’s custom instructions. |
| 2. Skill | Hermes / Cursor / agentskills.io | Load [`skills/ra-thor-employ/SKILL.md`](../skills/ra-thor-employ/SKILL.md). |
| 3. Local HTTP wrap | Developers | Run [`wrappers/local-shim/rathor_wrap.py`](../wrappers/local-shim/rathor_wrap.py) against *your* upstream. You hold the key. |
| 4. Lattice Chat local backend | Same device | `/chat.html` → Local Server → `http://localhost:11434/v1` (already ships). |

There is **no** public rathor.ai proxy that holds xAI / Anthropic / OpenAI keys. `app/api/grok/route.js` is research forest — do not treat it as a product.

## Custom-instruction snippets

- Grok: [`wrappers/custom-instructions/grok.md`](../wrappers/custom-instructions/grok.md)
- Claude: [`wrappers/custom-instructions/claude.md`](../wrappers/custom-instructions/claude.md)
- ChatGPT: [`wrappers/custom-instructions/chatgpt.md`](../wrappers/custom-instructions/chatgpt.md)
- Gemini: [`wrappers/custom-instructions/gemini.md`](../wrappers/custom-instructions/gemini.md)
- Cursor: [`wrappers/custom-instructions/cursor.md`](../wrappers/custom-instructions/cursor.md)

Same constitution. Different paste boxes.

## Local wrap (30 seconds)

```bash
export RATHOR_UPSTREAM=http://localhost:11434/v1
# optional: export RATHOR_UPSTREAM_KEY=sk-...
python3 wrappers/local-shim/rathor_wrap.py
# POST http://127.0.0.1:8787/v1/chat/completions
# GET  http://127.0.0.1:8787/v1/models
```

Point Lattice Chat or any OpenAI client at `http://127.0.0.1:8787/v1`. The shim injects the system constitution and labels the reply a draft. `stream=false` returns JSON. `stream=true` byte-forwards SSE (`text/event-stream`). `GET /v1/models` proxies upstream `/models` when it exists, else a one-item stub. There is **no** public rathor.ai key proxy.

### curl — stream false

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rathor-wrap","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

### curl — stream true

```bash
curl -N http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rathor-wrap","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

### Python OpenAI SDK — stream false / true

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="operator-held")

# stream false
print(
    client.chat.completions.create(
        model="rathor-wrap",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )
    .choices[0]
    .message.content
)

# stream true
for chunk in client.chat.completions.create(
    model="rathor-wrap",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
):
    piece = chunk.choices[0].delta.content
    if piece:
        print(piece, end="", flush=True)
```

## HOLD

- No hosted multi-tenant proxy.
- No “Ra-Thor mode certified by xAI / Anthropic / OpenAI.”
- No new default Cargo member.
- Combined AGSi stays SURMISE.
- Family walk unchanged.

Cursor seats for remaining code: [`docs/cursor-teams/WRAP_JOBS.md`](cursor-teams/WRAP_JOBS.md).

Thunder locked. yoi ⚡
