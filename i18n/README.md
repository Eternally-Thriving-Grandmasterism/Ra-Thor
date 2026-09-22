# Rathor.ai homepage locale packs

**Cache token:** `20260922b` (`i18n-chrome.js` and packs)  
**Source of truth:** `i18n/en.js`  
**Contact:** info@Rathor.ai

## Chrome-only policy (W4)

Offline packs translate **chrome**, not essays. Future copy edits to Employ / Privacy / FAQ **do not** demand 23 translations.

**Applied keys** (`js/i18n-chrome.js`):

- family labels — `navHome` … `navPrivacy`
- Follow — `followTitle`, `followX`, `followLinkedIn`, `followFacebook`
- hero / subhead — `headline`, `fusion`, `kicker`
- recent updates — `weekTitle`, `weekLineWrap`, `weekLineRa`, `weekLinePowrush`, `weekLineResearch`, `weekMore`
- session cards — titles, subtitles, CTAs (`grok*` `x*` `vibe*` `employCta`)
- Employ chrome — `employTitle`, `employSubtitle`
- Contact inquiry — `contactInquiry`
- Google tab chrome — `gTranslateBtn`, `gTranslateNote`

**Long copy stays English in git** (do not spend a seat translating these into 23 packs):

- Employ body (`article.rt-prose`)
- Privacy body
- FAQ answers (`faqA*`) and FAQ questions (`faqQ*`)

Missing key → English. Never blank. Never invent METR. Recent Updates lines are chrome, including `weekLineResearch`. The homepage site-lock still applies FAQ packs after chrome.

## Direction

`dir=rtl` only when the **applied string** for that node is actually RTL (`ar` / `fa` / `he` script). English fallback forces `dir=ltr` on that node. If most chrome nodes fell back, `<main>` stays `ltr`. `html[dir]` follows chrome only — not the language code alone. `<article>` / `.rt-prose` stay `dir=ltr lang=en` until a real translation exists. Family pill row stays Home…Privacy left-to-right (do not mirror).

Buttons on `index.html` load `/i18n/{lang}.js?v=20260922b`.

Service worker precaches `/i18n/*.js`. Offline packs are the product UX. Google Translate is a **new-tab URL** (`js/google-translate-optin.js`) — it does **not** inject `translate.google.com` (COEP `require-corp` would fail a widget). Do not relax COEP on `/chat.html` or worker paths.

| Code | Language | Notes |
| --- | --- | --- |
| en | English | Living voice 2026-09-15. Claim lock. Headline is the brand word. |
| ar | العربية | RTL chrome only, when the string is actually Arabic |
| es | Español | |
| fr | Français | faqA8 is NOT “RBE already here” |
| nl | Nederlands | |
| de | Deutsch | |
| zh | 简体中文 | |
| ja | 日本語 | |
| pt | Português | |
| ru | Русский | |
| hi | हिन्दी | |
| it | Italiano | |
| ko | 한국어 | |
| uk | Українська | |
| pl | Polski | |
| tr | Türkçe | |
| vi | Tiếng Việt | |
| id | Bahasa Indonesia | |
| sv | Svenska | |
| th | ไทย | |
| el | Ελληνικά | |
| fa | فارسی | RTL chrome only |
| he | עברית | RTL chrome only |

Do not copy the pre-2026-08-31 “RBE royalties dissolve into abundance” FAQ line. `faqA8` on every pack must stay: RBE is design intent, not a present economic fact. Packs may still contain FAQ essays historically; the site does not apply them.

AG-SML v1.1. Independent of xAI. Not certified aircraft, plants, chains, or a replaced money system.
