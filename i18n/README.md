# Rathor.ai homepage locale packs

**Cache token:** `20260923b` (`i18n-chrome.js`, `i18n-essay.js`, and packs)  
**Source of truth:** `i18n/en.js`  
**Contact:** info@Rathor.ai

## Visitor essays are packed (W4)

Offline packs apply **chrome** and **visitor essays**. Operator documents (`docs/**`) stay English.

**Chrome** (`js/i18n-chrome.js`):

- family labels — `navHome` … `navPrivacy`
- Follow — `followTitle`, `followX`, `followLinkedIn`, `followFacebook`
- hero / subhead — `headline`, `fusion`, `kicker`
- recent updates — `weekTitle`, `weekLineWrap`, `weekLineRa`, `weekLinePowrush`, `weekLineResearch`, `weekMore`
- session cards — titles, subtitles, CTAs (`grok*` `x*` `vibe*` `employCta`)
- Employ chrome — `employTitle`, `employSubtitle`
- install chip — `installTitle`, `installStatus`, `installCta`, `demoNote`
- Contact inquiry — `contactInquiry`
- Google tab chrome — `gTranslateBtn`, `gTranslateNote`

**Visitor essays** (`js/i18n-essay.js`, loaded after chrome):

- FAQ questions and answers (`faqQ*`, `faqA*`, `faqTitle`, `faqContact`)
- Employ article A–G (`article.rt-prose`)
- Privacy sections
- Briefing body
- Contact guidance and the commercial-inquiry form labels
- Launch, go-x, and science-watch short prose

FAQ keys that already had a translation still apply. New essay keys in this pass are **English copies** in the other 22 packs, so a missing translation never blanks the node. Real translation of those copies is a later seat. Do not treat an English copy as a finished translation.

Missing key → English. Never blank. Never invent METR. Recent Updates lines stay chrome, including `weekLineResearch`, and stay undated. `i18n-chrome.js` stamps family-nav and Follow labels, and mounts `#lang-selector` when a page has none. Site-lock delegates to chrome apply; essay apply follows that event. The family pill row and the language tabs stay left-to-right.

## Direction

`dir=rtl` only when the **applied string** for that node is actually RTL (`ar` / `fa` / `he` script). English fallback forces `dir=ltr lang=en` on that node. If most chrome nodes fell back, `<main>` stays `ltr`. `html[dir]` follows chrome only — not the language code alone, and not the essay. When a pack’s essay string differs from English, that `<article>` / `#faq` / `[data-rt-prose]` takes the applied language (`dir=rtl` only if the applied essay text is RTL script). An English copy keeps `dir=ltr lang=en`. Family pill row stays Home…Privacy left-to-right (do not mirror). Language tabs stay left-to-right.

Buttons on `index.html` load `/i18n/{lang}.js?v=20260923b`.

Service worker precaches every `/i18n/*.js` pack plus `/js/i18n-chrome.js` and `/js/i18n-essay.js`. Script fetches stay network-first. If the network fails, `/i18n/*.js`, `i18n-chrome.js`, and `i18n-essay.js` match the precache with the query string ignored, so `?v=20260923b` still opens offline. Offline packs are the product UX. Google Translate is a **new-tab URL** (`js/google-translate-optin.js`) — it does **not** inject `translate.google.com` (COEP `require-corp` would fail a widget). Do not relax COEP on `/chat.html` or worker paths.

| Code | Language | Notes |
| --- | --- | --- |
| en | English | Living voice 2026-09-15. Claim lock. Headline is the brand word. |
| ar | العربية | RTL only when the applied string is actually Arabic |
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
| fa | فارسی | RTL only when the applied string is actually Persian |
| he | עברית | RTL only when the applied string is actually Hebrew |

Do not copy the pre-2026-08-31 “RBE royalties dissolve into abundance” FAQ line. `faqA8` on every pack must stay: RBE is design intent, not a present economic fact. Essay apply fills a marked `faqA8` node from that pack line. The current homepage FAQ does not render question 8. The pack line must not state a resource-based economy as a present fact.

AG-SML v1.1. Independent of xAI. Not certified aircraft, plants, chains, or a replaced money system.
