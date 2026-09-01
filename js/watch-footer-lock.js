/* watch-footer-lock.js — insert Science watches into existing site footers
   Top family pills stay Home · Chat · Launch · Moments · Shard · Forge · Contact · Privacy.
   Contact: info@Rathor.ai
*/
(function () {
  if (typeof document === 'undefined') return;
  if (window.__rtWatchFooter) return;
  window.__rtWatchFooter = true;

  function insert() {
    var roots = document.querySelectorAll('.rt-site-footer, [data-rt-family-footer]');
    if (!roots.length) return;
    for (var r = 0; r < roots.length; r++) {
      if (roots[r].querySelector('a[href="/science-watches.html"]')) continue;
      var privacy = roots[r].querySelector('a[href="/privacy.html"]');
      if (!privacy || !privacy.parentNode) continue;
      var a = document.createElement('a');
      a.href = '/science-watches.html';
      a.textContent = 'Science watches';
      a.className = privacy.className || 'hover:text-amber-200';
      if (privacy.nextSibling) privacy.parentNode.insertBefore(a, privacy.nextSibling);
      else privacy.parentNode.appendChild(a);
    }
  }

  if (document.body) insert();
  else document.addEventListener('DOMContentLoaded', insert);
  window.addEventListener('load', insert);
  window.addEventListener('rathor-nav-ready', insert);
})();
