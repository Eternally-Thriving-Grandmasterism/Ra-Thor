/* HERO-SIZE-1: static check of the landing hero sizing rules. */
var fs = require('fs'), path = require('path'), assert = require('assert');
var html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!/min\(72vw, 16\.5rem\)/.test(html), 'old 16.5rem cap must be gone');
assert(html.indexOf('width: min(90vw, calc(min(78vh, 44rem) * 784 / 1168));') !== -1, 'mobile width rule');
assert(html.indexOf('width: min(100%, calc(min(78vh, 52rem) * 640 / 954));') !== -1, 'tablet width rule');
assert(html.indexOf('minmax(0, min(44%, calc(min(78vh, 46rem) * 640 / 954)))') !== -1, 'desktop column rule');
assert(/aspect-ratio: 784 \/ 1168;/.test(html), 'aspect ratio kept');
assert(/border-radius: 1\.15rem;/.test(html), 'rounded corners kept');
assert(/rgba\(240, 211, 106, 0\.38\)/.test(html), 'gold border kept');
console.log('hero-size-1 ok');
