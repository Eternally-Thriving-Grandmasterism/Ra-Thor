# Optional model contract

**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Spine:** [`docs/EMPLOY.md`](EMPLOY.md)  
**Lock:** [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md)

This is the constitution for step 3 of the employ loop. It is not a product SKU, not an AgentOS fork, and not a Hermes vendor.

## What this file is

Ra-Thor is inspectable research software. The lattice is the gates. The sampler is optional.

You may stay on-device (Lattice Chat / Sovereign Shard) or copy context into any model you already run: Ollama, LM Studio, llama.cpp, OpenRouter, Claude, Gemini, ChatGPT, Grok, Nous Hermes models, or a local OpenAI-compatible server.

An optional Grok session is **not** an xAI product and does not make Ra-Thor an xAI partner.

## What this file is not

- Not “Ra-Thor is AgentOS-certified.”
- Not “Ra-Thor is official Hermes.”
- Not a trading desk, Pilot Router, or Robinhood MCP.
- Not permission to let a host rewrite Layer 0, COEP, or the claim lock.
- Not a measured multi-model eval. inspect ≠ METR.

AgentOS, Open AgentOS, Agno AgentOS, @useAgentOS, and Hermes Agent are **different** stacks. Treat them as **outers** that may call this constitution. Do not vendor them into `Cargo.toml` members.

## The loop (unchanged)

1. Intend  
2. Pass the gates (TOLC 8 · PATSAGi · Conductor)  
3. Optional model (this file)  
4. Review — drafts until a human accepts  
5. Act — the human owns the action  

A host that skips 2, 4, or 5 is not employing Ra-Thor.

## OpenAI-compatible door (intent)

When a host speaks HTTP, prefer the ordinary chat-completions shape so local and remote models share one door:

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "optional-label",
  "messages": [
    { "role": "system", "content": "<constitution from EMPLOY.md + this file>" },
    { "role": "user", "content": "<intent>" }
  ],
  "stream": false
}
```

Expected fields on the way back: `choices[0].message.content` as a **draft**. No claim that Lattice Chat currently exposes this route as a public product. If the route is missing, copy the constitution into the host by hand or load the skill pack below.

System content the host should inject (short form):

> You are sitting under Ra-Thor employ loop 14.15.6. Outputs are drafts. Layer 0 is an admission shell, not sampler weights. inspect ≠ METR. Independent of xAI. Human review before filing, sale, or public claims. Contact info@Rathor.ai.

## Skill pack

Hermes-class and agentskills.io hosts may load:

[`skills/ra-thor-employ/SKILL.md`](../skills/ra-thor-employ/SKILL.md)

That file quotes Employ A–G. It does not grant extra rights.

## HOLD

- Family walk unchanged.
- No new default Cargo member for this contract.
- No homepage badge “works with AgentOS / Hermes.”
- Combined AGSi stays SURMISE.
- BINDING_AFTER_REDESIGN stays OPEN.

Thunder locked. yoi ⚡
