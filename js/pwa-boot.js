/* js/pwa-boot.js — parse-time boot. Must stay first in <head>, no defer.
 * Workspace 14.15.6 · info@Rathor.ai
 */
window.__rtPwa = window.__rtPwa || { ev: null };
window.addEventListener('beforeinstallprompt', function (e) {
  e.preventDefault();
  window.__rtPwa.ev = e;
});
try {
  document.documentElement.setAttribute('data-theme', localStorage.getItem('rathor-theme') || 'dark');
} catch (e) {
  document.documentElement.setAttribute('data-theme', 'dark');
}
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js', { scope: '/', updateViaCache: 'none' }).catch(function () {});
}
