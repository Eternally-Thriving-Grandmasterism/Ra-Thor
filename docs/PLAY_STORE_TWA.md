# Play Store after the PWA is installable

Workspace 14.15.6 · info@Rathor.ai

Do this only after Chrome installs Ra-Thor as a real WebAPK from rathor.ai (no Chrome badge).

## Path

1. Keep the live PWA criteria green: HTTPS, `manifest.json` with 192 + 512 PNG, `display: standalone`, service worker with a fetch handler.
2. Package a Trusted Web Activity with [PWABuilder](https://www.pwabuilder.com/) or Bubblewrap against `https://rathor.ai/`.
3. Add Digital Asset Links at `https://rathor.ai/.well-known/assetlinks.json` so the TWA opens the real site, not a Custom Tab.
4. Use the gold bolt 512 PNG as the Play icon. No JPEG favicon.
5. Store listing: independent lattice, offline Lattice Chat, no xAI affiliation claim.
6. Privacy: same policy as `/privacy.html` — no account we control.

Until that package exists, Chrome → rathor.ai → Install app is the official install.
