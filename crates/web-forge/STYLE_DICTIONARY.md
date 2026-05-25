# Style Dictionary Guide

This document explains how to use Style Dictionary with the design tokens in `web-forge`.

## Installation

```bash
npm install -D style-dictionary
```

## Basic Usage

1. Edit tokens in `design-tokens/tokens.json`
2. Run the build:

```bash
npx style-dictionary build --config ./style-dictionary.config.js
```

## Output

The config currently generates:
- `design-tokens/css/variables.css` — CSS Custom Properties

## Recommended Future Improvements

- Add Tailwind config transformer
- Add TypeScript definitions
- Support multiple themes (light/dark)

## Token Structure

We follow a 3-tier approach:

1. **Primitive** — Raw values
2. **Semantic** — Purpose-based names
3. **Component** — Specific to UI components (future)

This structure supports scalability and maintainability.