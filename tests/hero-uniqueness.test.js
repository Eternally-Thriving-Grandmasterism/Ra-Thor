/* RATHOR-ART-3: no two rathor.ai pages share a hero, and every referenced hero file exists. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

function identity(filename) {
  var stem = filename.replace(/\.[^.]+$/, '');
  stem = stem.replace(/-poster-\d+$/, '');
  stem = stem.replace(/-\d{3,4}$/, '');
  return stem;
}

function webpSize(buf) {
  assert(buf.slice(0, 4).toString() === 'RIFF' && buf.slice(8, 12).toString() === 'WEBP', 'not a webp');
  var kind = buf.slice(12, 16).toString();
  if (kind === 'VP8X') {
    return {
      w: 1 + buf.readUIntLE(24, 3),
      h: 1 + buf.readUIntLE(27, 3)
    };
  }
  if (kind === 'VP8 ') {
    var at = buf.indexOf(Buffer.from([0x9d, 0x01, 0x2a]));
    assert(at !== -1, 'VP8 start code missing');
    return {
      w: buf.readUInt16LE(at + 3) & 0x3fff,
      h: buf.readUInt16LE(at + 5) & 0x3fff
    };
  }
  if (kind === 'VP8L') {
    var b0 = buf[21], b1 = buf[22], b2 = buf[23], b3 = buf[24];
    return {
      w: 1 + (((b1 & 0x3f) << 8) | b0),
      h: 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6))
    };
  }
  throw new Error('unknown webp chunk ' + kind);
}

var HERO_URL = /\/assets\/art\/(hero-[A-Za-z0-9._-]+)/g;

function heroUrls(text) {
  var found = [];
  var m;
  HERO_URL.lastIndex = 0;
  while ((m = HERO_URL.exec(text))) found.push(m[1]);
  return found;
}

var skipDir = {
  '.git': 1,
  node_modules: 1,
  target: 1,
  dist: 1
};

function walkText(dir, out) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach(function (ent) {
    if (skipDir[ent.name]) return;
    var full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      walkText(full, out);
      return;
    }
    if (!/\.(html|js|css|json|txt|xml|webmanifest|yml|yaml|md)$/.test(ent.name)) return;
    out.push(path.relative(root, full));
  });
}

var textFiles = [];
walkText(root, textFiles);

var referenced = {};
textFiles.forEach(function (rel) {
  if (rel.indexOf('tests/') === 0) return;
  heroUrls(read(rel)).forEach(function (file) {
    referenced[file] = referenced[file] || [];
    referenced[file].push(rel);
  });
});

Object.keys(referenced).forEach(function (file) {
  var abs = path.join(root, 'assets/art', file);
  assert(fs.existsSync(abs), file + ' is referenced by ' + referenced[file].join(', ') + ' but missing');
  assert(fs.statSync(abs).size > 0, file + ' is empty');
});

var keptOlderHome = {
  'hero-home-rathor.mp4': 1,
  'hero-home-rathor-poster-480.webp': 1,
  'hero-home-rathor-poster-784.webp': 1
};
var onDisk = fs.readdirSync(path.join(root, 'assets/art')).filter(function (name) {
  return name.indexOf('hero-') === 0;
});
onDisk.forEach(function (file) {
  if (keptOlderHome[file]) return;
  assert(referenced[file], file + ' is on disk and unreferenced');
});

var pages = fs.readdirSync(root).filter(function (name) { return /\.html$/.test(name); });
var byIdentity = {};
pages.forEach(function (page) {
  var ids = {};
  heroUrls(read(page)).forEach(function (file) {
    ids[identity(file)] = 1;
  });
  Object.keys(ids).forEach(function (id) {
    if (!byIdentity[id]) byIdentity[id] = [];
    byIdentity[id].push(page);
  });
});
/* ART-UPLIFT-1: the only allowed sharing is one scene per page group, and the pages in a
 * group must differ by crop or tint (the figure class list). Exact page lists, nothing wider. */
var sharedGroup = {
  'hero-legal-hall-records': ['licence.html', 'third-party-models.html']
};
Object.keys(byIdentity).forEach(function (id) {
  if (sharedGroup[id]) {
    assert(byIdentity[id].slice().sort().join(',') === sharedGroup[id].slice().sort().join(','),
      id + ' may only be shared by ' + sharedGroup[id].join(', ') + ', found ' + byIdentity[id].join(', '));
    return;
  }
  assert(byIdentity[id].length === 1, id + ' is shared by ' + byIdentity[id].join(', '));
});

function figureClass(html) {
  var m = html.match(/<figure class="(rt-art-hero[^"]*)"/);
  assert(m, 'hero figure class missing');
  return m[1].split(/\s+/).sort().join(' ');
}
Object.keys(sharedGroup).forEach(function (id) {
  var seen = {};
  sharedGroup[id].forEach(function (page) {
    var cls = figureClass(read(page));
    assert(!seen[cls], page + ' must crop or tint ' + id + ' differently from ' + seen[cls]);
    seen[cls] = page;
  });
});

/* Generated docs pages share one cover scene, each with its own crop/tint. Only these four. */
var docsGen = require(path.join(root, 'scripts/docs-pretty/build.js'));
var docPages = docsGen.DOCS.map(function (d) { return d.html; });
assert(docPages.slice().sort().join(',') ===
  ['docs/CONTRIBUTIONS.html', 'docs/FAQ.html', 'docs/LINEAGE.html', 'docs/WHITEPAPER_v4.3.html'].join(','),
  'doc cover group is exactly the four styled docs');
var docCoverId = 'hero-docs-archive-light';
var docSeen = {};
docPages.forEach(function (rel) {
  var html = read(rel);
  var ids = heroUrls(html).map(identity);
  assert(ids.length && ids.every(function (id) { return id === docCoverId; }), rel + ' must use only ' + docCoverId);
  var bar = html.indexOf('class="rt-doc-bar"');
  var fig = html.indexOf('rt-doc-cover');
  var layout = html.indexOf('class="rt-doc-layout"');
  assert(bar !== -1 && bar < fig && fig < layout, rel + ' cover sits between .rt-doc-bar and .rt-doc-layout');
  var cls = figureClass(html);
  assert(!docSeen[cls], rel + ' must crop or tint the doc cover differently from ' + docSeen[cls]);
  docSeen[cls] = rel;
});
textFiles.forEach(function (rel) {
  /* WHITEPAPER-V43-1: docs/WHITEPAPER_v4.2.html is the frozen, byte-identical history page; it keeps the cover it shipped with. */
  if (rel.indexOf('tests/') === 0 || docPages.indexOf(rel) !== -1 || rel === 'scripts/docs-pretty/build.js' || rel === 'docs/WHITEPAPER_v4.2.html') return;
  assert(read(rel).indexOf(docCoverId) === -1, rel + ' must not reuse the doc cover');
});

var expectId = {
  'contact.html': 'hero-contact-comms-district',
  'micro-moment.html': 'hero-moment-research-garden',
  'pilot.html': 'hero-pilot-scoped-workbench',
  'examples.html': 'hero-examples-workbench-bolt',
  'lineage.html': 'hero-lineage-thread',
  'go-x.html': 'hero-gox-storm-signal',
  'licence.html': 'hero-legal-hall-records',
  'third-party-models.html': 'hero-legal-hall-records'
};
Object.keys(expectId).forEach(function (page) {
  assert(byIdentity[expectId[page]] && byIdentity[expectId[page]].indexOf(page) !== -1, page + ' must use ' + expectId[page]);
});

function figureImg(html) {
  var start = html.indexOf('class="rt-art-hero');
  assert(start !== -1, 'hero figure missing');
  var end = html.indexOf('</figure>', start);
  assert(end > start, 'hero figure unclosed');
  return html.slice(start, end);
}

var scenic = {
  /* ART-UPLIFT-2: the old (max-width: 900px) 192px pattern is gone, so phones pick the large file. */
  'contact.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'micro-moment.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 1080px) 100vw, 1024px' },
  'Launch-Ra-Thor.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 1080px) 100vw, 1024px' },
  'briefing.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'employ.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'privacy.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'chat.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 780px) 100vw, 736px' },
  'science-watches.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 1200px) 100vw, 1152px' },
  'sovereign-shard.html': { w: 1500, h: 844, load: 'high', sizes: '(max-width: 1200px) 100vw, 1152px' },
  'web-forge.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 1200px) 100vw, 1152px' },
  'constellation-week.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 720px) 100vw, 672px' },
  'thanks.html': { w: 1672, h: 941, load: 'high', sizes: '(max-width: 620px) 100vw, 576px' },
  'pilot.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'examples.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'lineage.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'go-x.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 720px) 100vw, 640px' },
  'licence.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'third-party-models.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 800px) 100vw, 768px' },
  'docs/WHITEPAPER_v4.3.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 840px) 100vw, 800px' },
  'docs/FAQ.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 840px) 100vw, 800px' },
  'docs/LINEAGE.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 840px) 100vw, 800px' },
  'docs/CONTRIBUTIONS.html': { w: 1280, h: 720, load: 'high', sizes: '(max-width: 840px) 100vw, 800px' }
};
Object.keys(scenic).forEach(function (page) {
  var spec = scenic[page];
  var fig = figureImg(read(page));
  assert(fig.indexOf('sizes="' + spec.sizes + '"') !== -1, page + ' sizes');
  assert(fig.indexOf('decoding="async"') !== -1, page + ' decoding');
  assert(fig.indexOf('width="' + spec.w + '"') !== -1 && fig.indexOf('height="' + spec.h + '"') !== -1, page + ' width/height');
  if (spec.load === 'lazy') {
    assert(fig.indexOf('loading="lazy"') !== -1, page + ' must lazy-load like the other below-fold scenic heroes');
  } else {
    assert(fig.indexOf('fetchpriority="high"') !== -1, page + ' must fetch the lead scenic hero first');
    assert(fig.indexOf('loading="lazy"') === -1, page + ' lead hero must not lazy-load');
  }
  var alt = fig.match(/\balt="([^"]*)"/);
  assert(alt && alt[1].trim().length > 20, page + ' needs a descriptive alt');
  var srcset = fig.match(/\bsrcset="([^"]*)"/);
  assert(srcset, page + ' srcset');
  srcset[1].split(',').forEach(function (part) {
    var m = part.trim().match(/\/assets\/art\/(\S+)\s+(\d+)w/);
    assert(m, page + ' srcset entry ' + part);
    assert(m[1].indexOf('-' + m[2] + '.') !== -1, page + ' srcset width must match the filename');
    var buf = fs.readFileSync(path.join(root, 'assets/art', m[1]));
    var dim = webpSize(buf);
    assert(dim.w === Number(m[2]), m[1] + ' pixel width ' + dim.w + ' != ' + m[2]);
    if (Number(m[2]) === spec.w) assert(dim.h === spec.h, m[1] + ' pixel height');
  });
});

var homeHtml = read('index.html');
assert(homeHtml.indexOf('/assets/art/rathor-hero-thunder-stand.jpg') !== -1, 'home uses the thunder stand poster');
assert(homeHtml.indexOf('rathor-hero-living-thunder') === -1, 'home no longer points at the living thunder backup');
assert(homeHtml.indexOf('hero-home-rathor') === -1, 'home no longer points at the older portrait');
pages.forEach(function (page) {
  if (page === 'index.html') return;
  assert(read(page).indexOf('rathor-hero-living-thunder') === -1, page + ' must keep its own hero');
  assert(read(page).indexOf('rathor-hero-thunder-stand') === -1, page + ' must keep its own hero');
});
var homeAlt = figureImg(homeHtml).match(/\balt="([^"]*)"/)[1];
assert(homeAlt.indexOf('black and antique-gold armour') !== -1, 'home alt names the armour');
assert(homeAlt.indexOf('winged gold helm') !== -1, 'home alt names the helm');
assert(homeAlt.indexOf('glowing sun chest disc') !== -1, 'home alt names the chest disc');
assert(homeAlt.indexOf('Eye of Horus shoulder seals') !== -1, 'home alt names the shoulder seals');
assert(homeAlt.indexOf('shield') === -1, 'home alt must not name a shield');
assert(homeAlt.indexOf('lightning-marked hammer') !== -1, 'home alt names the hammer');
assert(homeAlt.indexOf('stormy peak') !== -1, 'home alt names the storm');

[
  'assets/art/hero-contact-relay-station-768.webp',
  'assets/art/hero-contact-relay-station-1120.webp',
  'assets/art/hero-moment-field-lab-768.webp',
  'assets/art/hero-moment-field-lab-1060.webp',
  'assets/art/hero-gox-beacon-768.webp',
  'assets/art/hero-gox-beacon-1672.webp',
  'assets/art/hero-pilot-citadel-arrival-768.webp',
  'assets/art/hero-pilot-citadel-arrival-1500.webp'
].forEach(function (rel) {
  assert(!fs.existsSync(path.join(root, rel)), rel + ' must be removed');
});

var oldHits = [];
textFiles.forEach(function (rel) {
  if (rel === 'tests/hero-uniqueness.test.js') return;
  var text = read(rel);
  if (/hero-contact-relay-station|hero-moment-field-lab|hero-gox-beacon|hero-pilot-citadel-arrival/.test(text)) {
    oldHits.push(rel);
  }
});
assert(oldHits.length === 0, 'old relay-station / field-lab / gox-beacon / citadel-arrival heroes still named in ' + oldHits.join(', '));

/* ART-UPLIFT-3: pilot gets new scoped-workbench art (Sherif 8:41 PM) and examples a canon redo (hammer bolt on both faces).
   EXAMPLES-NO-HAMMER-1 (Sherif 5:10 PM ET, Nov 25): the examples hero has the hammer removed, so its byte pins moved. */
var supplied = {
  'assets/art/hero-contact-comms-district-1672.webp': 244690,
  'assets/art/hero-contact-comms-district-768.webp': 76200,
  'assets/art/hero-moment-research-garden-1672.webp': 245016,
  'assets/art/hero-moment-research-garden-768.webp': 82366,
  'assets/art/hero-docs-archive-light-1280.webp': 157536,
  'assets/art/hero-docs-archive-light-768.webp': 73704,
  'assets/art/hero-examples-workbench-bolt-1280.webp': 131722,
  'assets/art/hero-examples-workbench-bolt-768.webp': 58164,
  'assets/art/hero-gox-storm-signal-1280.webp': 147906,
  'assets/art/hero-gox-storm-signal-768.webp': 70348,
  'assets/art/hero-legal-hall-records-1280.webp': 91930,
  'assets/art/hero-legal-hall-records-768.webp': 36706,
  'assets/art/hero-lineage-thread-1280.webp': 110842,
  'assets/art/hero-lineage-thread-768.webp': 47158,
  'assets/art/hero-pilot-scoped-workbench-1280.webp': 136092,
  'assets/art/hero-pilot-scoped-workbench-768.webp': 58760
};
Object.keys(supplied).forEach(function (rel) {
  assert(fs.statSync(path.join(root, rel)).size === supplied[rel], rel + ' must stay the supplied file');
});

console.log('hero-uniqueness.test.js ok (' + Object.keys(byIdentity).length + ' unique page heroes)');
