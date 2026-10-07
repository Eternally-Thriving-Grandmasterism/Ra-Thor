/* Home X reel — stills first. Each muted mp4 is attached only when the
 * shared hero gate allows it and the frame is near the viewport.
 * The same gate covers the two home-card clips.
 * Without this script the pictures remain.
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';

  var CLIPS = {
    sentinel: '/assets/art/reel-sentinel-architecture.mp4',
    tolc: '/assets/art/reel-tolc-heart.mp4',
    hammer: '/assets/art/reel-rathor-winged-hammer.mp4',
    handshake: '/assets/art/reel-grok-handshake.mp4',
    build: '/assets/art/reel-grok-build.mp4'
  };

  function shared() {
    if (typeof window !== 'undefined' && window.rtMutedClip && typeof window.rtMutedClip.heroVideoAllowed === 'function') {
      return window.rtMutedClip;
    }
    if (typeof require === 'function') return require('./home-hero.js');
    return null;
  }

  function reelClipShouldPlay(env, near) {
    var api = shared();
    if (!api || typeof api.heroVideoAllowed !== 'function') return false;
    return api.heroVideoAllowed(env) === true && near === true;
  }

  if (typeof window !== 'undefined') window.rtReelClipShouldPlay = reelClipShouldPlay;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { reelClipShouldPlay: reelClipShouldPlay, CLIPS: CLIPS };
  }

  function posterFor(figure) {
    var still = figure.querySelector('img');
    if (still && still.currentSrc) return still.currentSrc;
    if (still && still.getAttribute('src')) return still.getAttribute('src');
    return '';
  }

  function syncClip(figure, near) {
    var api = shared();
    if (!api) return;
    var video = figure.querySelector('video');
    if (!video) return;
    var env = api.readEnv();
    if (!api.heroVideoAllowed(env)) {
      api.release(video);
      return;
    }
    if (near !== true) {
      api.pause(video);
      return;
    }
    if (video.getAttribute('data-rt-live') === '1') {
      api.silence(video);
      var pending = video.play();
      if (pending && typeof pending.catch === 'function') pending.catch(function () {});
      return;
    }
    api.bind(video, CLIPS[figure.getAttribute('data-rt-reel')], posterFor(figure));
  }

  function boot() {
    if (typeof IntersectionObserver !== 'function' || !document.querySelectorAll) return;
    var frames = document.querySelectorAll('[data-rt-reel]');
    if (!frames.length) return;
    var near = [];
    var io = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];
        var idx = Number(entry.target.getAttribute('data-rt-reel-i'));
        near[idx] = entry.isIntersecting === true;
        syncClip(entry.target, near[idx]);
      }
    }, { root: null, rootMargin: '200px 0px', threshold: 0 });

    function resync() {
      for (var i = 0; i < frames.length; i++) syncClip(frames[i], near[i] === true);
    }

    var api = shared();
    for (var n = 0; n < frames.length; n++) {
      frames[n].setAttribute('data-rt-reel-i', String(n));
      near[n] = false;
      io.observe(frames[n]);
      (function (video) {
        if (!video || !api) return;
        video.addEventListener('volumechange', function () {
          if (!video.muted || video.volume !== 0) api.silence(video);
        });
      })(frames[n].querySelector('video'));
    }
    if (api && typeof api.watchEnv === 'function') api.watchEnv(resync);
  }

  if (typeof document === 'undefined' || !document.getElementById) return;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
