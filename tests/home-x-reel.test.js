/* RATHOR-VIDEO-1: three steward clips under the home intro.
 * Stills are the default. js/home-x-reel.js attaches muted mp4s
 * through the shared home-hero gate, and only when a frame is near view.
 */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var files = {
  'assets/art/reel-sentinel-architecture.mp4': 843892,
  'assets/art/reel-sentinel-architecture-poster.webp': 36244,
  'assets/art/reel-tolc-heart.mp4': 1472913,
  'assets/art/reel-tolc-heart-poster.webp': 53848,
  'assets/art/reel-rathor-winged-hammer.mp4': 1397223,
  'assets/art/reel-rathor-winged-hammer-poster.webp': 58124
};
Object.keys(files).forEach(function (rel) {
  var full = path.join(root, rel);
  assert(fs.existsSync(full), rel + ' must exist');
  assert(fs.statSync(full).size === files[rel], rel + ' must stay the supplied ' + files[rel] + '-byte file');
});

var html = read('index.html');
assert(html.indexOf('reel-sentinel-architecture.mp4') === -1, 'index must not embed the sentinel mp4');
assert(html.indexOf('reel-tolc-heart.mp4') === -1, 'index must not embed the tolc mp4');
assert(html.indexOf('reel-rathor-winged-hammer.mp4') === -1, 'index must not embed the hammer mp4');
assert(html.indexOf('2078966782883135582') === -1, 'hammer post must not be linked; its text is not repeated');
assert(html.indexOf('/js/home-x-reel.js') !== -1, 'index must load home-x-reel.js');
assert(html.indexOf('/js/home-hero.js') < html.indexOf('/js/home-x-reel.js'), 'hero gate script must load before the reel');

var heroStart = html.indexOf('id="rt-home-hero"');
var heroEnd = html.indexOf('</figure>', heroStart);
assert(heroStart !== -1 && heroEnd > heroStart, 'home hero figure must stay');
var hero = html.slice(heroStart, heroEnd);
assert(hero.indexOf('reel-') === -1, 'home hero figure must not gain the reel');
assert(hero.indexOf('hero-home-rathor-poster-784.webp') !== -1, 'home hero still must stay');

var reelStart = html.indexOf('id="rt-x-reel"');
var reelEnd = html.indexOf('</section>', reelStart);
assert(reelStart !== -1 && reelEnd > reelStart, 'reel section must exist');
assert(html.indexOf('id="steward-line"') < reelStart && reelStart < html.indexOf('id="lang-selector"'), 'reel sits under the intro and above the language row');
var reel = html.slice(reelStart, reelEnd);
assert(reel.indexOf('data-i18n') === -1, 'reel copy stays verbatim and is not an i18n key');
assert(reel.indexOf('From <a href="https://x.com/AlphaProMega" rel="noopener" target="_blank">@AlphaProMega</a> on X') !== -1, 'heading stays plain');
assert(reel.indexOf('Eternal Sentinel Architecture v8.0 visualized') !== -1, 'sentinel caption stays verbatim');
assert(reel.indexOf('https://x.com/AlphaProMega/status/2001770410904317989') !== -1, 'sentinel caption links to its post');
assert(reel.indexOf('for all creations and creatures, to thrive with positive emotions forever and ever') !== -1, 'tolc caption stays verbatim');
assert(reel.indexOf('https://x.com/AlphaProMega/status/2057902030614601883') !== -1, 'tolc caption links to its post');
assert(reel.indexOf('rel="noopener"') !== -1 && reel.indexOf('target="_blank"') !== -1, 'caption links open safely');

var alts = [
  ['464', '688', 'A luminous blue-white humanoid figure stands between a bright sun and a spiral galaxy, inside glowing circles and geometric marks.'],
  ['640', '952', 'A blue lightning-lined robot figure holds out an open hand under a glowing pink heart with the word TOLC.'],
  ['960', '644', 'A dark hammer with fiery wings and lightning sits in front of a glowing ring, above fiery title text reading RA-THOR.']
];
alts.forEach(function (row) {
  assert(reel.indexOf('width="' + row[0] + '"') !== -1 && reel.indexOf('height="' + row[1] + '"') !== -1, row[0] + ' poster must set width and height');
  assert(reel.indexOf('alt="' + row[2] + '"') !== -1, 'alt must describe the picture');
});
assert((reel.match(/loading="lazy"/g) || []).length === 3, 'each poster img is lazy');
assert((reel.match(/<figcaption\b/g) || []).length === 2, 'only the two captioned posts have figcaptions');
assert(reel.indexOf('data-rt-reel="hammer"') !== -1, 'hammer frame must be present');
var hammerAt = reel.indexOf('data-rt-reel="hammer"');
var hammerFig = reel.slice(hammerAt, reel.indexOf('</figure>', hammerAt));
assert(hammerFig.indexOf('<figcaption') === -1, 'hammer has no caption');

var videos = reel.match(/<video\b[^>]*>/g);
assert(videos && videos.length === 3, 'three video shells');
videos.forEach(function (tag) {
  assert(tag.indexOf('muted') !== -1, 'video must be muted');
  assert(tag.indexOf('autoplay') !== -1, 'video must autoplay');
  assert(tag.indexOf('loop') !== -1, 'video must loop');
  assert(tag.indexOf('playsinline') !== -1, 'video must be playsinline');
  assert(tag.indexOf('preload="none"') !== -1, 'video preload must be none');
  assert(tag.indexOf('aria-hidden="true"') !== -1, 'video duplicates the alt and must be aria-hidden');
  assert(tag.indexOf('controls') === -1, 'video must not show controls');
});
assert(reel.indexOf('<source') === -1 && reel.indexOf('video/mp4') === -1, 'static reel must not include an mp4 source');
assert(html.indexOf('@media (prefers-reduced-motion: reduce)') !== -1 && html.indexOf('.rt-x-frame video { display: none !important; }') !== -1, 'reduced motion hides reel video');

var js = read('js/home-x-reel.js');
assert(js.indexOf('reel-sentinel-architecture.mp4') !== -1, 'script must name the sentinel mp4');
assert(js.indexOf('reel-tolc-heart.mp4') !== -1, 'script must name the tolc mp4');
assert(js.indexOf('reel-rathor-winged-hammer.mp4') !== -1, 'script must name the hammer mp4');
assert(js.indexOf('heroVideoAllowed') !== -1, 'reel must use the shared hero gate');
assert(js.indexOf('slow-2g') === -1 && js.indexOf('saveData') === -1, 'reel must not copy the gate conditions');
assert(js.indexOf('IntersectionObserver') !== -1, 'reel must watch viewport nearness');
assert(js.indexOf("rootMargin: '200px 0px'") !== -1, 'near view uses a 200px root margin');
assert(js.indexOf('api.pause(video)') !== -1, 'leaving view pauses');
assert(js.indexOf('api.release(video)') !== -1, 'a closed gate drops the source');
assert(js.indexOf('api.watchEnv(resync)') !== -1, 'reel follows the hero env watch');
assert(js.indexOf('still.currentSrc') !== -1, 'poster must reuse the still that already loaded');

var heroApi = require(path.join(root, 'js/home-hero.js'));
var reelApi = require(path.join(root, 'js/home-x-reel.js'));
function deny(env, near, label) {
  assert(reelApi.reelClipShouldPlay(env, near) === false, label);
}
function allow(env, near, label) {
  assert(reelApi.reelClipShouldPlay(env, near) === true, label);
}
var wide = { reducedMotion: false, narrow: false, saveData: false, effectiveType: '4g' };
deny(Object.assign({}, wide, { reducedMotion: true }), true, 'reduced motion blocks video');
deny(Object.assign({}, wide, { saveData: true }), true, 'saveData blocks video');
deny(Object.assign({}, wide, { effectiveType: 'slow-2g' }), true, 'slow-2g blocks video');
deny(Object.assign({}, wide, { effectiveType: '2g' }), true, '2g blocks video');
deny(Object.assign({}, wide, { effectiveType: '3g' }), true, '3g blocks video');
deny(Object.assign({}, wide, { narrow: true }), true, 'narrow viewport blocks video');
deny(wide, false, 'offscreen blocks video even on a fast network');
allow(wide, true, '4g wide and near allows video');
allow(Object.assign({}, wide, { effectiveType: '' }), true, 'unknown network and near allows video');
assert(heroApi.pause && heroApi.release && heroApi.bind && heroApi.watchEnv, 'shared clip helpers stay on the hero module');

['sw.js', 'public/sw.js', 'offline.html'].forEach(function (rel) {
  var text = read(rel);
  Object.keys(files).forEach(function (media) {
    assert(text.indexOf(path.basename(media)) === -1, rel + ' must not precache ' + media);
  });
  assert(text.indexOf('home-x-reel') === -1, rel + ' must not reference the reel script');
});

['contact.html', 'micro-moment.html', 'pilot.html', 'Launch-Ra-Thor.html', 'constellation-week.html'].forEach(function (rel) {
  var page = read(rel);
  assert(page.indexOf('rt-x-reel') === -1 && page.indexOf('reel-sentinel-architecture') === -1, rel + ' must not take the reel');
});

var headers = read('_headers');
assert(headers.indexOf('media-src') === -1, '_headers must not send a media-src that would block same-origin clips');
assert(html.indexOf('Content-Security-Policy') === -1, 'index must not add a CSP meta that blocks self media');

console.log('home-x-reel.test.js ok');
