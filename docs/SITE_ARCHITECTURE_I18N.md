# Site architecture — modular i18n (root-cause fix)

**Contact:** info@Rathor.ai  
**Date:** 2026-07-20

## Problem

The legacy `index.html` embedded all 11 language dictionaries in one ~75KB file. Connector / agent tool limits (and practical edit safety) made reliable single-shot minting or full-file replace unreliable. Editing “slice by slice” still required re-uploading the entire file each time.

## Solution

| Path | Role | Typical size |
|------|------|----------------|
| `index.html` | Shell markup + i18n **loader** runtime | ~25KB |
| `i18n/<lang>.js` | One language dictionary | ~2–8KB each |
| `site-updates-v14.js` | Optional progressive DOM patcher (legacy path) | ~10KB |

Languages load on demand via `/i18n/<lang>.js`. Changing Arabic no longer requires rewriting English FAQ HTML.

## Loader contract

Each `i18n/<lang>.js` must assign:

```js
window.translations = window.translations || {};
window.translations["en"] = { /* keys */ };
```

The shell calls `loadLangScript(lang)` then `applyTranslations(lang)`.

## Editing policy going forward

1. **Copy / structure** → edit `index.html` (now under size limits).
2. **One language’s strings** → edit only `i18n/<lang>.js`.
3. **Never** re-inline all dictionaries into `index.html`.

## Deploy note

Ensure the host serves `/i18n/*.js` with correct MIME (`application/javascript` or `text/javascript`). GitHub Pages does this by default.

**Thunder locked in.**
