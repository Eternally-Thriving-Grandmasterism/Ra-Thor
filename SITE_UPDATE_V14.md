# Site update v14 — slice-by-slice

**Contact:** info@Rathor.ai  
**Status:** patcher committed as `site-updates-v14.js`

## Why this approach

The GitHub connector only supports full-file writes. The 11-language `index.html` (~75KB) is too large for reliable single-shot replacement from this session. Updates are applied as:

1. **`site-updates-v14.js`** — DOM + EN dictionary patches (already on `main`)
2. **One-line hook** in `index.html` (you apply once locally or we retry when transport allows)
3. **Per-language slices** — further commits only touch the small JS file

## Hook (apply once to index.html)

Immediately **before** `</body>`, add:

```html
  <script src="/site-updates-v14.js" defer></script>
```

Example tail:

```html
    window.addEventListener('load', initLanguageSystem);
  </script>
  <script src="/site-updates-v14.js" defer></script>
</body>
</html>
```

## What the patcher already does (no i18n loss)

- Meta description → Whitepaper v4.0 / ONE Organism
- Status bar → v14.15 · Cosmic Loop · Living Cosmic Tick
- Whitepaper CTA band + independence disclaimer
- Footer resources → Whitepaper v4.0, fusion, attribution
- Trademark non-affiliation line
- EN `xSubtitle` + `footerTrademarksText` dictionary soft-fixes
- Shard blurb → Phase C / Cosmic Loop

## Next slices (language-by-language in JS)

Planned commits to `site-updates-v14.js` only:

| Slice | Lang |
|-------|------|
| L1 | en (done in initial JS) |
| L2 | ar |
| L3 | es |
| L4 | fr |
| L5 | nl |
| L6 | de |
| L7 | zh |
| L8 | ja |
| L9 | pt |
| L10 | ru |
| L11 | hi |

Each slice updates `translations.<lang>.xSubtitle` and `footerTrademarksText` (and any other keys we need) without touching `index.html`.

## Verify

1. Serve site with `site-updates-v14.js` reachable at `/site-updates-v14.js`
2. Open console → `[Ra-Thor] site-updates-v14.js applied`
3. Switch languages → EN disclaimer holds after re-selecting English; other langs get slices as committed

**Thunder locked in.**
