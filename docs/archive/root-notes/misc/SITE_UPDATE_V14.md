# Site update v14 — modular architecture (root-cause fixed)

**Contact:** info@Rathor.ai  
**Status:** modular i18n on `main`

## Root cause

Legacy `index.html` inlined **all 11 languages** (~75KB). Tool/connector hard limits made single-shot minting unreliable. Partial edits still required full-file re-upload.

## Fix

| Path | Role |
|------|------|
| `index.html` | Shell + loader only (~25KB) |
| `i18n/<lang>.js` | One language each (~2–8KB) |
| `docs/SITE_ARCHITECTURE_I18N.md` | Architecture policy |
| `site-updates-v14.js` | Optional progressive patcher |

## Languages on disk

- `i18n/en.js` ✅
- `i18n/ar.js` ✅
- `i18n/es.js` ✅
- `i18n/fr.js` ✅
- `i18n/zh.js` ✅
- `i18n/ja.js` ✅
- Remaining (`nl`, `de`, `pt`, `ru`, `hi`) — land in follow-up commits or copy from local `/tmp/i18n` if mid-session

## Deploy

1. Host must serve `/i18n/*.js`
2. Open site → language tabs load modules on demand
3. Edit one language without touching the shell

**Never re-inline all dictionaries into index.html.**

Thunder locked in.
