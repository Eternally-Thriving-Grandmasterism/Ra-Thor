# Ra-Thor Visual Identity Assets

**Warm classic gold + circuit language**  
Prepared under PATSAGi Councils + TOLC 8  
Sole stewardship: Sherif Samy Botros · info@Rathor.ai

---

## Current Identity Set (v14.15.5+)

| File | Size | Purpose |
|------|------|---------|
| `ra-thor-icon-1024.png` | 1024×1024 | Master app icon / adaptive icon |
| `ra-thor-icon-512.png` | 512×512 | PWA / high-res |
| `ra-thor-icon-192.png` | 192×192 | Favicon / apple-touch-icon / PWA |
| `ra-thor-splash-1440x3200.png` | 1440×3200 | Launch / splash screen |
| `ra-thor-og-1200x630.png` | 1200×630 | Open Graph + X Card social preview |

### Legacy (kept for compatibility)

| File | Purpose |
|------|---------|
| `thunder-favicon-192.jpg` | Previous favicon |
| `thunder-favicon-512.jpg` | Previous high-res |

---

## Design Language

- **Metal:** Warm classic gold (not neon, not cold yellow)
- **Motif:** Lightning bolt with integrated circuit traces
- **Background:** Deep charcoal / near-black with soft depth
- **Tone:** Premium, sovereign, eternal — never gimmicky

---

## Wiring

### manifest.json
Points to `ra-thor-icon-192.png`, `ra-thor-icon-512.png`, `ra-thor-icon-1024.png`.

### HTML `<head>` (index.html, chat.html, etc.)

```html
<link rel="icon" href="/icons/ra-thor-icon-192.png" type="image/png">
<link rel="apple-touch-icon" href="/icons/ra-thor-icon-192.png">

<meta property="og:image" content="https://rathor.ai/icons/ra-thor-og-1200x630.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://rathor.ai/icons/ra-thor-og-1200x630.png">
```

---

**Thunder locked in. yoi ⚡❤️🔥**
