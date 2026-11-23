/* LOOP-XFADE-1: crossfade the end of an opted-in looping <video data-loop-xfade>
 * back into its start. A muted clone is made only when the clip is in view and
 * near its end, fades in over the loop point, then is removed. No video files
 * change. Reduced motion or Save-Data: plain loop as before. Vanilla, no deps.
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';
  var FADE = 0.7, LEAD = 2.5;
  function mq(q) { try { return window.matchMedia(q).matches; } catch (e) { return false; } }
  function off() {
    var c = navigator.connection || {};
    return mq('(prefers-reduced-motion: reduce)') || !!c.saveData;
  }
  function drop(s) {
    if (!s.clone) return;
    s.clone.pause();
    s.clone.removeAttribute('src');
    try { s.clone.load(); } catch (e) {}
    if (s.clone.parentNode) s.clone.parentNode.removeChild(s.clone);
    s.clone = null; s.fading = false;
  }
  function make(v, s) {
    var c = document.createElement('video'), cs = getComputedStyle(v);
    c.muted = true; c.defaultMuted = true; c.playsInline = true;
    c.setAttribute('muted', ''); c.setAttribute('playsinline', '');
    c.setAttribute('aria-hidden', 'true'); c.tabIndex = -1;
    c.preload = 'auto';
    c.style.cssText = 'display:block;position:absolute;margin:0;pointer-events:none;opacity:0;' +
      'left:' + v.offsetLeft + 'px;top:' + v.offsetTop + 'px;width:' + v.offsetWidth + 'px;height:' +
      v.offsetHeight + 'px;object-fit:' + cs.objectFit + ';object-position:' + cs.objectPosition +
      ';background:' + cs.backgroundColor + ';z-index:' + (cs.zIndex === 'auto' ? 'auto' : cs.zIndex) +
      ';transition:opacity ' + FADE + 's linear';
    c.src = v.currentSrc;
    v.parentNode.insertBefore(c, v.nextSibling);
    s.clone = c;
  }
  function tick(v, s) {
    var d = v.duration, t = v.currentTime;
    if (!s.seen || off() || v.paused || !v.currentSrc || !isFinite(d) || d < FADE * 4) { if (!s.fading) drop(s); return; }
    if (!s.clone && t >= d - LEAD) make(v, s);
    if (s.clone && !s.fading && t >= d - FADE) {
      s.fading = true;
      var c = s.clone;
      try { c.currentTime = 0; } catch (e) {}
      var p = c.play();
      if (p && p.catch) p.catch(function () { drop(s); });
      c.style.opacity = '1';
      setTimeout(function () {
        if (s.clone !== c) return;
        var done = function () {
          v.removeEventListener('seeked', done);
          c.style.transition = 'opacity .3s linear'; c.style.opacity = '0';
          setTimeout(function () { if (s.clone === c) drop(s); }, 340);
        };
        v.addEventListener('seeked', done);
        try { v.currentTime = c.currentTime; } catch (e) { done(); }
      }, FADE * 1000 + 60);
    }
  }
  function wire(v) {
    if (v.getAttribute('data-loop-xfade-on')) return;
    v.setAttribute('data-loop-xfade-on', '1');
    var s = { seen: false, clone: null, fading: false, held: false };
    v.addEventListener('timeupdate', function () { tick(v, s); });
    v.addEventListener('emptied', function () { drop(s); });
    v.addEventListener('pause', function () { if (!s.fading) drop(s); });
    if (typeof IntersectionObserver !== 'function') { s.seen = true; return; }
    new IntersectionObserver(function (es) {
      s.seen = es[es.length - 1].isIntersecting;
      if (!s.seen) {
        drop(s);
        if (!v.paused) { s.held = true; v.pause(); }
      } else if (s.held) {
        s.held = false;
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
      }
    }).observe(v);
  }
  function boot() {
    var list = document.querySelectorAll('video[loop][data-loop-xfade]');
    for (var i = 0; i < list.length; i++) wire(list[i]);
  }
  if (typeof document === 'undefined' || !document.querySelectorAll) return;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
