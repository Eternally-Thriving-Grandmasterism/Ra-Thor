/* ORIGIN-STORY-1: assert the three origin-story paragraphs are present verbatim in index.html */
var fs = require('fs');
var path = require('path');
var assert = require('assert');

var html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

var P1 = 'Ra-Thor grew out of building Powrush, the game universe Autonomicity Games Inc. has developed since 2016, and out of a problem Sherif Botros saw firsthand: AI that hallucinated too often to trust. On November 18, 2025, he began focusing fully on turning his AlphaProMega quad+Check and Anti-Hallucination Protocols (AHP) into systems that AIs like Grok could actually use. By early December he was sharing the AlphaProMega Suite of Protocols openly on X, inviting people to try it and refining it quickly from their feedback.';
var P2 = "The work kept evolving. In Sherif's own records, the NEXi-AGi repo followed on January 19, 2026. It was renamed Rathor-NEXi, then Ra-Thor, and remade from MercyOS-Pinnacle. Along the way NEXi became one module of something larger, and Ra-Thor went live at rathor.ai at the start of February 2026.";
var P3 = 'Ra-Thor comes from Autonomicity Games Inc., and Ra-Thor now helps build Powrush-MMO.';

assert(html.indexOf(P1) !== -1, 'ORIGIN-STORY-1 P1 must be present verbatim in index.html');
assert(html.indexOf(P2) !== -1, 'ORIGIN-STORY-1 P2 must be present verbatim in index.html');
assert(html.indexOf(P3) !== -1, 'ORIGIN-STORY-1 P3 must be present verbatim in index.html');

assert(html.indexOf('How Ra-Thor Began') !== -1, 'ORIGIN-STORY-1 subhead must be present in index.html');

console.log('PASS origin-story-1: all three paragraphs and subhead present verbatim');
