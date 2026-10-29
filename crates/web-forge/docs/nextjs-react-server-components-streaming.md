# React Server Components + Streaming Integration with web-forge

This guide explains how to effectively use **React Server Components (RSC)** and **Streaming** when building a frontend that integrates with a `web-forge` Rust backend.

## Why RSC + Streaming?

- Reduces client-side JavaScript
- Enables progressive rendering and faster perceived performance
- Naturally supports calling backend services (like `web-forge`)
- Works excellently with Suspense for loading states

## Recommended Architecture

```
Next.js App Router (RSC + Streaming)
        │
        ├── Server Components
        │       └── Call Rust backend (web-forge)
        │
        ├── Suspense Boundaries
        │       └── Stream when orchestration completes
        │
        └── Client Components ( Islands )
                └── Only hydrate interactive parts
```

## Practical Pattern

### 1. Server Component Calling web-forge

```tsx
async function WebForgeSection({ prompt }: { prompt: string }) {
  const res = await fetch('http://localhost:3001/orchestrate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  const report = await res.json();

  return (
    <div>
      <h2>Generated Report</h2>
      <pre>{JSON.stringify(report, null, 2)}</pre>
    </div>
  );
}
```

### 2. Using Suspense for Streaming

```tsx
import { Suspense } from 'react';

import { WebForgeSection } from './WebForgeSection';

export default function Dashboard() {
  return (
    <div>
      <h1>Dashboard</h1>

      <Suspense fallback={<div>Loading orchestration result...</div>}>
        <WebForgeSection prompt="Create an accessible hero section" />
      </Suspense>
    </div>
  );
}
```

## Benefits for web-forge Integration

- You can stream fast parts of the UI while `web-forge` performs complex orchestration
- Reduces time-to-first-byte impact of heavy backend processing
- Clean separation: Rust handles logic, Next.js handles UI and streaming
- Excellent developer experience with the App Router

## Best Practices

- Wrap slow `web-forge` calls in `Suspense`
- Use multiple Suspense boundaries for parallel streaming
- Keep Client Components minimal (islands)
- Pass only necessary data from Rust to the frontend
- Leverage `OrchestrationReport` as the primary data contract

## Trade-offs

- Adds some complexity compared to simple SSR
- Requires careful cache and streaming configuration
- Debugging streaming issues can be more involved

## Summary

React Server Components + Streaming is currently one of the best ways to build a frontend that consumes a powerful Rust backend like `web-forge`. It offers strong performance characteristics while maintaining excellent developer experience.
