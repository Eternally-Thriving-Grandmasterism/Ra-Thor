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

var onDisk = fs.readdirSync(path.join(root, 'assets/art')).filter(function (name) {
  return name.indexOf('hero-') === 0;
});
onDisk.forEach(function (file) {
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
Object.keys(byIdentity).forEach(function (id) {
  assert(byIdentity[id].length === 1, id + ' is shared by ' + byIdentity[id].join(', '));
});

var expectId = {
  'contact.html': 'hero-contact-comms-district',
  'micro-moment.html': 'hero-moment-research-garden',
  'pilot.html': 'hero-pilot-citadel-arrival',
  'index.html': 'hero-home-rathor'
};
Object.keys(expectId).forEach(function (page) {
  assert(byIdentity[expectId[page]] && byIdentity[expectId[page]][0] === page, page + ' must use ' + expectId[page]);
});

function figureImg(html) {
  var start = html.indexOf('class="rt-art-hero');
  assert(start !== -1, 'hero figure missing');
  var end = html.indexOf('</figure>', start);
  assert(end > start, 'hero figure unclosed');
  return html.slice(start, end);
}

var scenic = {
  'contact.html': { w: 1672, h: 941, load: 'lazy' },
  'micro-moment.html': { w: 1672, h: 941, load: 'high' },
  'pilot.html': { w: 1500, h: 844, load: 'high' }
};
Object.keys(scenic).forEach(function (page) {
  var spec = scenic[page];
  var fig = figureImg(read(page));
  assert(fig.indexOf('sizes="(max-width: 900px) 192px, 1400px"') !== -1, page + ' sizes');
  assert(fig.indexOf('decoding="async"') !== -1, page + ' decoding');
  assert(fig.indexOf('width="' + spec.w + '"') !== -1 && fig.indexOf('height="' + spec.h + '"') !== -1, page + ' width/height');
  if (spec.load === 'lazy') {
    assert(fig.indexOf('loading="lazy"') !== -1, page + ' must lazy-load like the other below-fold scenic heroes');
  } else {
    assert(fig.indexOf('fetchpriority="high"') !== -1, page + ' must fetch the lead scenic hero first');
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

var homeAlt = figureImg(read('index.html')).match(/\balt="([^"]*)"/)[1];
assert(homeAlt.indexOf('rests a hammer head-down on the ground') !== -1, 'home alt must say the hammer rests');
assert(homeAlt.indexOf('raises') === -1, 'home alt must not say the hammer is raised');

[
  'assets/art/hero-contact-relay-station-768.webp',
  'assets/art/hero-contact-relay-station-1120.webp',
  'assets/art/hero-moment-field-lab-768.webp',
  'assets/art/hero-moment-field-lab-1060.webp'
].forEach(function (rel) {
  assert(!fs.existsSync(path.join(root, rel)), rel + ' must be removed');
});

var oldHits = [];
textFiles.forEach(function (rel) {
  if (rel === 'tests/hero-uniqueness.test.js') return;
  var text = read(rel);
  if (text.indexOf('hero-contact-relay-station') !== -1 || text.indexOf('hero-moment-field-lab') !== -1) {
    oldHits.push(rel);
  }
});
assert(oldHits.length === 0, 'old relay-station / field-lab heroes still named in ' + oldHits.join(', '));

var supplied = {
  'assets/art/hero-contact-comms-district-1672.webp': 244690,
  'assets/art/hero-contact-comms-district-768.webp': 76200,
  'assets/art/hero-moment-research-garden-1672.webp': 245016,
  'assets/art/hero-moment-research-garden-768.webp': 82366,
  'assets/art/hero-pilot-citadel-arrival-1500.webp': 241698,
  'assets/art/hero-pilot-citadel-arrival-768.webp': 86132
};
Object.keys(supplied).forEach(function (rel) {
  assert(fs.statSync(path.join(root, rel)).size === supplied[rel], rel + ' must stay the supplied file');
});

console.log('hero-uniqueness.test.js ok (' + Object.keys(byIdentity).length + ' unique page heroes)');
