/* family-nav-2026-08-22.js
   Shared public-surface network for rathor.ai
   Workspace 14.15.6 · Lattice Chat v14.18.x
   Contact: info@Rathor.ai
*/
(function () {
  if (window.__rtFamilyNav) return;
  window.__rtFamilyNav = true;

  var LINKS = [
    { href: '/', label: 'Home' },
    { href: '/chat.html', label: 'Chat' },
    { href: '/Launch-Ra-Thor.html', label: 'Launch' },
    { href: '/sovereign-shard.html', label: 'Shard' },
    { href: '/web-forge.html', label: 'Forge' },
    { href: '/contact.html', label: 'Contact' },
    { href: '/privacy.html', label: 'Privacy' }
  ];

  function norm(p) {
    if (!p || p === '') return '/';
    p = p.split('?')[0].split('#')[0];
    if (p.charAt(p.length - 1) === '/' && p.length > 1) p = p.slice(0, -1);
    if (p === '/index.html') return '/';
    return p;
  }

  var here = norm(location.pathname || '/');

  function isHere(href) {
    return norm(href) === here;
  }

  function bar(kind) {
    var nav = document.createElement('nav');
    nav.id = kind === 'top' ? 'rt-family-nav' : 'rt-family-footer';
    nav.setAttribute('aria-label', 'Ra-Thor family');
    nav.style.cssText = kind === 'top'
      ? 'position:sticky;top:0;z-index:40;display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:center;align-items:center;padding:0.45rem 0.7rem;background:rgba(0,0,0,0.88);border-bottom:1px solid rgba(252,211,77,0.28);backdrop-filter:blur(8px);'
      : 'display:flex;flex-wrap:wrap;gap:0.55rem;justify-content:center;align-items:center;padding:0.7rem;border-top:1px solid rgba(252,211,77,0.2);margin-top:1rem;';

    LINKS.forEach(function (item) {
      var a = document.createElement('a');
      a.href = item.href;
      a.textContent = item.label;
      a.style.cssText = 'font:600 11px/1.2 system-ui,sans-serif;padding:0.35rem 0.65rem;border-radius:999px;text-decoration:none;border:1px solid rgba(252,211,77,0.35);color:#fde68a;';
      if (isHere(item.href)) {
        a.setAttribute('aria-current', 'page');
        a.style.background = '#fbbf24';
        a.style.color = '#000';
        a.style.borderColor = '#fbbf24';
      }
      nav.appendChild(a);
    });
    return nav;
  }

  function mount() {
    if (!document.getElementById('rt-family-nav')) {
      var top = bar('top');
      document.body.insertBefore(top, document.body.firstChild);
    }
    if (!document.getElementById('rt-family-footer')) {
      document.body.appendChild(bar('footer'));
    }
  }

  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
