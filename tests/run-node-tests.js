#!/usr/bin/env node
/* Run every tests/*.test.js in its own node process. Honor tests/QUARANTINE.md. */
var fs = require('fs');
var path = require('path');
var cp = require('child_process');

var testsDir = __dirname;
var quarantinePath = path.join(testsDir, 'QUARANTINE.md');

function parseQuarantine(content) {
  var map = Object.create(null);
  content.split('\n').forEach(function (line) {
    var trimmed = line.trim();
    if (!trimmed || trimmed.charAt(0) === '#') return;
    var sep = trimmed.indexOf(' — ');
    if (sep === -1) {
      throw new Error('QUARANTINE.md lines must be "<filename> — <reason>": ' + trimmed);
    }
    var file = trimmed.slice(0, sep).trim();
    var reason = trimmed.slice(sep + 3).trim();
    if (!file || !reason) {
      throw new Error('QUARANTINE.md lines must be "<filename> — <reason>": ' + trimmed);
    }
    map[file] = reason;
  });
  return map;
}

function unknownQuarantineEntries(quarantine, testFileNames) {
  return Object.keys(quarantine).filter(function (f) {
    return testFileNames.indexOf(f) === -1;
  });
}

function main() {
  var quarantine = fs.existsSync(quarantinePath)
    ? parseQuarantine(fs.readFileSync(quarantinePath, 'utf8'))
    : {};

  Object.keys(quarantine)
    .sort()
    .forEach(function (file) {
      console.log('QUARANTINED: ' + file + ' — ' + quarantine[file]);
    });

  var files = fs
    .readdirSync(testsDir)
    .filter(function (f) {
      return f.endsWith('.test.js');
    })
    .sort();

  var failed = false;

  files.forEach(function (file) {
    var filePath = path.join(testsDir, file);
    var result = cp.spawnSync(process.execPath, [filePath], {
      stdio: 'inherit',
      env: process.env,
    });
    var passed = result.status === 0;
    var reason = quarantine[file];
    if (reason) {
      if (passed) {
        console.error('STALE QUARANTINE (test now passes; remove from QUARANTINE.md): ' + file);
        failed = true;
      } else {
        console.log('quarantined failure (expected): ' + file);
      }
    } else if (!passed) {
      console.error('FAILED: ' + file);
      failed = true;
    }
  });

  unknownQuarantineEntries(quarantine, files).forEach(function (f) {
    console.error('UNKNOWN QUARANTINE ENTRY: ' + f);
    failed = true;
  });

  process.exit(failed ? 1 : 0);
}

if (require.main === module) {
  main();
}

module.exports = { parseQuarantine, unknownQuarantineEntries };
