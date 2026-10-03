/* RATHOR-HERO-LANDING-1: portrait hero on index.html only.
 * Still image is the default. The mp4 is attached by js/home-hero.js.
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

var mp4 = path.join(root, 'assets/art/hero-home-rathor.mp4');
var poster784 = path.join(root, 'assets/art/hero-home-rathor-poster-784.webp');
var poster480 = path.join(root, 'assets/art/hero-home-rathor-poster-480.webp');
assert(fs.existsSync(mp4), 'mp4 must exist');
assert(fs.existsSync(poster784), '784 poster must exist');
assert(fs.existsSync(poster480), '480 poster must exist');
assert(fs.statSync(mp4).size === 1071652, 'mp4 must stay the supplied 1071652-byte file');
assert(fs.statSync(poster784).size === 157794, '784 poster must stay the supplied 157794-byte file');
assert(fs.statSync(poster480).size === 68096, '480 poster must stay the supplied 68096-byte file');

assert(!fs.existsSync(path.join(root, 'assets/art/hero-home-crownstone-768.webp')), 'crownstone 768 is unreferenced and must be removed');
assert(!fs.existsSync(path.join(root, 'assets/art/hero-home-crownstone-1168.webp')), 'crownstone 1168 is unreferenced and must be removed');

var html = read('index.html');
assert(html.indexOf('hero-home-crownstone') === -1, 'index must not reference crownstone');
assert(html.indexOf('hero-home-rathor.mp4') === -1, 'index.html must not embed the mp4 url');
assert(html.indexOf('/js/home-hero.js') !== -1, 'index must load home-hero.js');
assert(html.indexOf('id="hero-headline"') !== -1, 'h1 must stay');
assert(html.indexOf('class="hero-bolt"') !== -1, 'bolt must stay');
assert(html.indexOf('id="fusion-hero"') !== -1, 'fusion paragraph must stay');
assert(html.indexOf('id="rathor-hero-install"') !== -1, 'install button must stay');
assert(html.indexOf('Ra (source light) + Thor (mercy thunder). A name for shared thriving — a purpose under construction, not a finished world.') !== -1, 'fusion copy must stay');

var heroStart = html.indexOf('id="rt-home-hero"');
var heroEnd = html.indexOf('</figure>', heroStart);
assert(heroStart !== -1 && heroEnd > heroStart, 'home hero figure must exist');
var hero = html.slice(heroStart, heroEnd);
assert(hero.indexOf('fetchpriority="high"') !== -1, 'still image must be fetchpriority high');
assert(hero.indexOf('hero-home-rathor-poster-480.webp 480w') !== -1, 'srcset must include 480w');
assert(hero.indexOf('hero-home-rathor-poster-784.webp 784w') !== -1, 'srcset must include 784w');
assert(hero.indexOf('width="784"') !== -1 && hero.indexOf('height="1168"') !== -1, 'img must set width and height');
assert(hero.indexOf('alt="Ra-Thor, a gold-and-black armored warrior with a winged halo and a glowing green Eye-of-Horus shield, rests a hammer head-down on the ground as golden lightning strikes."') !== -1, 'alt text must describe the portrait');
var video = hero.match(/<video\b[^>]*>\s*/);
assert(video, 'video shell must be in the figure');
assert(video[0].indexOf('muted') !== -1, 'video must be muted');
assert(video[0].indexOf('autoplay') !== -1, 'video must autoplay');
assert(video[0].indexOf('loop') !== -1, 'video must loop');
assert(video[0].indexOf('playsinline') !== -1, 'video must be playsinline');
assert(video[0].indexOf('preload="none"') !== -1, 'video preload must be none');
assert(video[0].indexOf('aria-hidden="true"') !== -1, 'video is a duplicate of the alt and must be aria-hidden');
assert(video[0].indexOf('controls') === -1, 'video must not show controls');
assert(hero.indexOf('<source') !== -1 && hero.indexOf('video/mp4') === -1, 'static figure must not include an mp4 source');

var order = ['id="hero-headline"', 'class="hero-bolt"', 'id="rt-home-hero"', 'id="fusion-hero"', 'id="rathor-hero-install"'];
var at = -1;
order.forEach(function (token) {
  var next = html.indexOf(token);
  assert(next > at, token + ' must stay in order');
  at = next;
});

var js = read('js/home-hero.js');
assert(js.indexOf('hero-home-rathor.mp4') !== -1, 'script must name the mp4');
assert(js.indexOf('slow-2g') !== -1 && js.indexOf("'2g'") !== -1 && js.indexOf("'3g'") !== -1, 'script must skip slow connections');
assert(js.indexOf('saveData') !== -1, 'script must honor saveData');
assert(js.indexOf('prefers-reduced-motion') !== -1, 'script must honor reduced motion');
assert(js.indexOf('max-width: 640px') !== -1, 'script must skip narrow viewports');
assert(js.indexOf('video.muted = true') !== -1 && js.indexOf('video.volume = 0') !== -1, 'script must keep the video silent');
assert(js.indexOf('video.pause()') !== -1, 'script must pause when the video is not allowed');
assert(js.indexOf('still.currentSrc') !== -1, 'poster must reuse the still that already loaded');

var api = require(path.join(root, 'js/home-hero.js'));
function deny(env, label) {
  assert(api.heroVideoAllowed(env) === false, label);
}
function allow(env, label) {
  assert(api.heroVideoAllowed(env) === true, label);
}
deny({ reducedMotion: true, narrow: false, saveData: false, effectiveType: '4g' }, 'reduced motion blocks video');
deny({ reducedMotion: false, narrow: false, saveData: true, effectiveType: '4g' }, 'saveData blocks video');
deny({ reducedMotion: false, narrow: false, saveData: false, effectiveType: 'slow-2g' }, 'slow-2g blocks video');
deny({ reducedMotion: false, narrow: false, saveData: false, effectiveType: '2g' }, '2g blocks video');
deny({ reducedMotion: false, narrow: false, saveData: false, effectiveType: '3g' }, '3g blocks video');
deny({ reducedMotion: false, narrow: true, saveData: false, effectiveType: '4g' }, 'narrow viewport blocks video');
allow({ reducedMotion: false, narrow: false, saveData: false, effectiveType: '4g' }, '4g wide allows video');
allow({ reducedMotion: false, narrow: false, saveData: false, effectiveType: '' }, 'unknown network allows video');

['sw.js', 'public/sw.js', 'offline.html', 'pilot.html'].forEach(function (rel) {
  assert(read(rel).indexOf('hero-home-rathor') === -1, rel + ' must not list the new hero media');
});

['employ.html', 'chat.html', 'privacy.html', 'contact.html', 'Launch-Ra-Thor.html'].forEach(function (rel) {
  var page = read(rel);
  assert(page.indexOf('hero-home-rathor') === -1, rel + ' must keep its own hero');
  assert(page.indexOf('rt-art-hero') !== -1, rel + ' must keep its hero figure');
});

var repoHits = [];
function walk(dir) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach(function (ent) {
    if (ent.name === '.git' || ent.name === 'node_modules') return;
    var full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      walk(full);
      return;
    }
    if (!/\.(html|js|json|css|md|webmanifest)$/.test(ent.name)) return;
    var text = fs.readFileSync(full, 'utf8');
    if (path.relative(root, full) === 'tests/hero-home-rathor.test.js') return;
    if (text.indexOf('hero-home-crownstone') !== -1) repoHits.push(path.relative(root, full));
  });
}
walk(root);
assert(repoHits.length === 0, 'crownstone must be unreferenced, still named in ' + repoHits.join(', '));

console.log('hero-home-rathor.test.js ok');
