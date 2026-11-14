#!/usr/bin/env python3
"""CLAIMS-TONE-3 mechanical markdown sweep.

Reproducible edits, and nothing else:

1. Non-archive ``.md`` only, longest exact-case pattern first:
   - ``Ra-Thor AGI framework`` -> ``Ra-Thor framework``
   - ``Ra-Thor AGI`` -> ``Ra-Thor``
   - ``AGI lattice`` -> ``lattice``
   Then fix only a doubled word or an article the deletion itself made unfit
   (``an lattice`` -> ``a lattice``).

2. Retitle the exact heading/bold-title string
   ``Artificial Godly Superintelligence aspiration Roadmap``
   to exactly ``AGi (Artificial Godly Intelligence) Roadmap``
   in ``.md`` files, archive included. Direct quotes are not edited.
   The one live heading sits inside a ```markdown copy-paste fence; that
   heading line is still retitled. No other line inside a fence is edited
   by this step.

3. Prepend the CLAIMS-TONE-2 historical-record banner to archive ``.md``
   files that do not already contain it, byte-for-byte. Those files get
   no other edit.

``AGI`` is the uppercase token. A letter, digit, underscore, or hyphen on
either side keeps the token (so ``AGi``, ``PATSAGi``, ``QSA-AGi``,
``NEXi-AGi``, ``AGSi``, ``QSA-AGI``, and ``AGI-guided`` stay).

Run from anywhere. Idempotent. ``--dry-run`` prints the report and writes
nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNER = (
    "> **Historical record.** This archived document reflects earlier wording "
    "and is not the living claim ceiling. `PUBLIC_CLAIM.lock.md` governs "
    "current public claims.\n\n"
)

TWEET_REL = "docs/Philosophy/TOLC-Flame-Eternal-Verified-Tweet-TOLC-2026.md"

OLD_TITLE = "Artificial Godly Superintelligence aspiration Roadmap"
NEW_TITLE = "AGi (Artificial Godly Intelligence) Roadmap"

MERCY_SENTENCE = "makes careful, mercy-evaluated decisions"

# Exact-case, longest first. Hyphen is part of a compound, not a boundary.
RX_FRAMEWORK = re.compile(
    r"(?<![A-Za-z0-9_])Ra-Thor AGI framework(?![A-Za-z0-9_-])"
)
RX_RATHOR = re.compile(r"(?<![A-Za-z0-9_])Ra-Thor AGI(?![A-Za-z0-9_-])")
RX_LATTICE = re.compile(r"(?<![A-Za-z0-9_-])AGI lattice(?![A-Za-z0-9_-])")
PATTERNS = (
    ("Ra-Thor AGI framework", RX_FRAMEWORK, "Ra-Thor framework"),
    ("Ra-Thor AGI", RX_RATHOR, "Ra-Thor"),
    ("AGI lattice", RX_LATTICE, "lattice"),
)

RX_AGI_TOKEN = re.compile(r"(?<![A-Za-z0-9_])AGI(?![A-Za-z0-9_])")
RX_DOUBLED = re.compile(r"\b([A-Za-z]+(?:-[A-Za-z]+)*)\s+\1\b")
RX_URL = re.compile(r"https?://[^\s<>)\]]+|www\.[^\s<>)\]]+")
RX_AUTOLINK = re.compile(r"<(?:https?://|mailto:)[^>\s]+>", re.IGNORECASE)
RX_HTML_ATTR = re.compile(
    r"(?i)\b(?:href|src)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)"
)
RX_ATTR = re.compile(
    r"(?i)\b(?:said|says|wrote|writes|posted|tweeted|according to|quoted by)\b"
    r"|[—–]\s*(?:Sherif|@[A-Za-z0-9_]|[A-Z][a-z]+)"
    r"|\b(?:tweet|twitter\.com|x\.com/)\b"
)
RX_HEADING = re.compile(r"^( {0,3})(#{1,6})([ \t]+)(.*)$")
RX_BOLD_TITLE = re.compile(
    r"^( {0,3})(\*\*|__)(.+)(\*\*|__)[ \t]*$"
)
PROTECTED_TOKENS = ("AGi", "PATSAGi", "QSA-AGi", "NEXi-AGi", "AGSi")
LOCK_NAME_RX = re.compile(
    r"(?i)(?:^|[_\-.])lock(?:[_\-.]|$)|(?:^|[_\-.])lock\.md$|\.lock\.md$"
)
LOCK_SENTINELS = ("locked verbatim", "The text above is locked")


def rel_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_archive(rel: str) -> bool:
    return "/archive/" in f"/{rel}" or rel.startswith("archive/")


def is_tweet(rel: str) -> bool:
    return rel == TWEET_REL


def filename_locked(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    if name == "PUBLIC_CLAIM.lock.md" or name == "GLOSSARY.md":
        return True
    return LOCK_NAME_RX.search(name) is not None


def content_locked(text: str) -> bool:
    head = text[:4000]
    return any(s in head for s in LOCK_SENTINELS)


def iter_md():
    for path in ROOT.rglob("*.md"):
        if not path.is_file():
            continue
        rel = rel_of(path)
        if "/.git/" in f"/{rel}":
            continue
        yield path, rel


def covered(pos: int, spans: list[tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= pos < end:
            return True
        if start > pos:
            break
    return False


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for start, end in spans[1:]:
        ps, pe = out[-1]
        if start <= pe:
            out[-1] = (ps, max(pe, end))
        else:
            out.append((start, end))
    return out


def fence_spans(text: str) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    offsets = []
    off = 0
    for line in lines:
        offsets.append(off)
        off += len(line)
    spans = []
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^( {0,3})(`{3,}|~{3,})", lines[i])
        if not m:
            i += 1
            continue
        ch = m.group(2)[0]
        flen = len(m.group(2))
        start = offsets[i]
        j = i + 1
        close = None
        while j < n:
            m2 = re.match(
                rf"^( {{0,3}})({re.escape(ch)}{{{flen},}})[ \t]*\r?\n?$",
                lines[j],
            )
            if m2 and set(m2.group(2)) == {ch}:
                close = offsets[j] + len(lines[j])
                break
            j += 1
        if close is None:
            spans.append((start, len(text)))
            break
        spans.append((start, close))
        i = j + 1
    return spans


def inline_code_spans(text: str, fences: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    i = 0
    n = len(text)
    fi = 0
    while i < n:
        while fi < len(fences) and fences[fi][1] <= i:
            fi += 1
        if fi < len(fences) and fences[fi][0] <= i < fences[fi][1]:
            i = fences[fi][1]
            continue
        if text[i] != "`":
            i += 1
            continue
        j = i
        while j < n and text[j] == "`":
            j += 1
        ticks = j - i
        k = j
        found = None
        while k < n:
            while fi < len(fences) and fences[fi][1] <= k:
                fi += 1
            if fi < len(fences) and fences[fi][0] <= k < fences[fi][1]:
                k = fences[fi][1]
                continue
            if text[k] != "`":
                k += 1
                continue
            m = k
            while m < n and text[m] == "`":
                m += 1
            if m - k == ticks:
                found = m
                break
            k = m
        if found is None:
            i = j
            continue
        spans.append((i, found))
        i = found
    return spans


def link_destination_spans(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    for rx in (RX_URL, RX_AUTOLINK, RX_HTML_ATTR):
        for m in rx.finditer(text):
            if not covered(m.start(), blocked):
                spans.append((m.start(), m.end()))
    i = 0
    n = len(text)
    while True:
        j = text.find("](", i)
        if j < 0:
            break
        if covered(j, blocked):
            i = j + 2
            continue
        k = j + 2
        depth = 1
        while k < n and depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            elif text[k] == "\n":
                break
            k += 1
        if depth == 0:
            spans.append((j + 1, k))
            i = k
        else:
            i = j + 2
    i = 0
    while True:
        j = text.find("][", i)
        if j < 0:
            break
        if covered(j, blocked):
            i = j + 2
            continue
        k = text.find("]", j + 2)
        if k < 0 or "\n" in text[j + 2 : k]:
            i = j + 2
            continue
        spans.append((j + 1, k + 1))
        i = k + 1
    for m in re.finditer(r"(?m)^ {0,3}\[[^\]]+\]:[ \t]*\S.*$", text):
        if not covered(m.start(), blocked):
            spans.append((m.start(), m.end()))
    return spans


def quote_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for m in re.finditer(r"“[^”]*”|\"[^\"\n]*\"", text):
        spans.append((m.start(), m.end()))
    return spans


def attributed_quote_spans(text: str, quotes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Quotes with an attribution on the same line, or within one short clause.

    A sign-off a paragraph later (``— @Name``) does not make an image prompt
    someone else's words. The verified tweet file is skipped as a whole.
    """
    kept = []
    for start, end in quotes:
        before = text[max(0, start - 80) : start]
        if before.count("\n") > 0:
            before = before.rsplit("\n", 1)[-1]
        newline = text.find("\n", end)
        after = text[end : newline if newline >= 0 else end + 60]
        after = after[:60]
        if RX_ATTR.search(before) or RX_ATTR.search(after):
            kept.append((start, end))
    return kept


def blockquote_post_spans(text: str) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    offsets = []
    off = 0
    for line in lines:
        offsets.append(off)
        off += len(line)
    spans = []
    i = 0
    n = len(lines)
    while i < n:
        if not re.match(r" {0,3}>", lines[i]):
            i += 1
            continue
        j = i + 1
        while j < n and (
            re.match(r" {0,3}>", lines[j]) or lines[j].strip() == ""
        ):
            if lines[j].strip() == "" and (
                j + 1 >= n or not re.match(r" {0,3}>", lines[j + 1])
            ):
                break
            j += 1
        block = "".join(lines[i:j])
        if re.search(r"(?i)twitter\.com|x\.com/|tweeted|^\s*>\s*tweet\b", block):
            end = offsets[j] if j < n else len(text)
            spans.append((offsets[i], end))
        i = j
    return spans


def epithet_masks(text: str) -> list[tuple[int, int]]:
    fences = fence_spans(text)
    code = inline_code_spans(text, fences)
    blocked = merge_spans(fences + code)
    links = link_destination_spans(text, blocked)
    quotes = quote_spans(text)
    attributed = attributed_quote_spans(text, quotes)
    posts = blockquote_post_spans(text)
    return merge_spans(blocked + links + attributed + posts)


def apply_epithets(text: str) -> tuple[str, Counter, list[str]]:
    masks = epithet_masks(text)
    counts: Counter = Counter()
    notes: list[str] = []
    rebuilt = []
    cursor = 0
    for start, end in masks + [(len(text), len(text))]:
        if cursor < start:
            before = text[cursor:start]
            after = before
            for _name, rx, repl in PATTERNS:
                found = rx.findall(after)
                counts[_name] += len(found)
                after = rx.sub(repl, after)
            fixed = fix_grammar(before, after)
            if fixed != after:
                notes.append(fixed)
            rebuilt.append(fixed)
        if start < end:
            rebuilt.append(text[start:end])
        cursor = end
    return "".join(rebuilt), counts, notes


def fix_grammar(before: str, after: str) -> str:
    if before == after:
        return after
    if "an lattice" not in before and "An lattice" not in before:
        after = re.sub(r"\bAn lattice\b", "A lattice", after)
        after = re.sub(r"\ban lattice\b", "a lattice", after)

    def repl(m: re.Match) -> str:
        if m.group(0) in before:
            return m.group(0)
        return m.group(1)

    prev = None
    while prev != after:
        prev = after
        after = RX_DOUBLED.sub(repl, after)
    return after


def retitle(text: str) -> tuple[str, int]:
    # Fences stay editable so a ```markdown heading can be retitled.
    # Inline code, link targets, and direct quotes stay protected.
    fences = fence_spans(text)
    quotes = quote_spans(text)
    code = inline_code_spans(text, fences)
    links = link_destination_spans(text, merge_spans(fences + code))
    blocked = merge_spans(quotes + code + links)
    lines = text.splitlines(keepends=True)
    out = []
    count = 0
    off = 0
    for line in lines:
        raw = line[:-1] if line.endswith("\n") else line
        nl = "\n" if line.endswith("\n") else ""
        if raw.endswith("\r"):
            raw = raw[:-1]
            nl = "\r" + nl
        replaced, n = retitle_line(raw, off, blocked)
        count += n
        out.append(replaced + nl)
        off += len(line)
    return "".join(out), count


def occurrence_blocked(line_off: int, local: int, blocked: list[tuple[int, int]]) -> bool:
    return covered(line_off + local, blocked)


def retitle_line(raw: str, line_off: int, blocked: list[tuple[int, int]]) -> tuple[str, int]:
    heading = RX_HEADING.match(raw)
    bold = RX_BOLD_TITLE.match(raw)
    if heading:
        body = heading.group(4)
        idx = body.find(OLD_TITLE)
        if idx < 0:
            return raw, 0
        prefix_len = len(heading.group(1) + heading.group(2) + heading.group(3))
        if occurrence_blocked(line_off, prefix_len + idx, blocked):
            return raw, 0
        body = body.replace(OLD_TITLE, NEW_TITLE, 1)
        return (
            heading.group(1) + heading.group(2) + heading.group(3) + body,
            1,
        )
    if bold and bold.group(2) == bold.group(4):
        inner = bold.group(3)
        idx = inner.find(OLD_TITLE)
        if idx < 0:
            return raw, 0
        prefix_len = len(bold.group(1) + bold.group(2))
        if occurrence_blocked(line_off, prefix_len + idx, blocked):
            return raw, 0
        # A bold title is the line, not a label plus a sentence.
        if len(inner.strip()) > len(OLD_TITLE) + 40 and inner.strip() != OLD_TITLE:
            stripped = inner.strip()
            if stripped != OLD_TITLE and not stripped.startswith(OLD_TITLE):
                return raw, 0
        inner = inner.replace(OLD_TITLE, NEW_TITLE, 1)
        return bold.group(1) + bold.group(2) + inner + bold.group(4), 1
    return raw, 0


def literal_count(text: str, token: str) -> int:
    n = 0
    start = 0
    while True:
        i = text.find(token, start)
        if i < 0:
            return n
        n += 1
        start = i + len(token)


def classify_agi(text: str, rel: str) -> tuple[Counter, list[tuple[str, str]], Counter]:
    reasons: Counter = Counter()
    samples: list[tuple[str, str]] = []
    left_words: Counter = Counter()
    fences = fence_spans(text)
    code = inline_code_spans(text, fences)
    blocked_code = merge_spans(fences + code)
    links = link_destination_spans(text, blocked_code)
    quotes = quote_spans(text)
    attributed = merge_spans(attributed_quote_spans(text, quotes))
    posts = merge_spans(blockquote_post_spans(text))
    for m in RX_AGI_TOKEN.finditer(text):
        pos = m.start()
        line_start = text.rfind("\n", 0, pos) + 1
        line_end = text.find("\n", m.end())
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end]
        prev = text[pos - 1] if pos else ""
        nxt = text[m.end()] if m.end() < len(text) else ""
        if is_tweet(rel):
            reason = "verified tweet file left unchanged (Sherif's post)"
        elif filename_locked(rel) or content_locked(text):
            reason = "locked file"
        elif covered(pos, fences):
            reason = "inside a fenced code block"
        elif covered(pos, code):
            reason = "inside inline code"
        elif covered(pos, links):
            reason = "inside a link target, URL, anchor, or reference"
        elif covered(pos, attributed) or covered(pos, posts):
            reason = "inside an attributed quote, third-party quote, or post blockquote"
        elif prev == "-" or nxt == "-":
            reason = "hyphenated compound (AGI is not a free token)"
        elif re.search(
            r"AGI Lattice|AGI lattices|Ra-Thor AGI Framework|Ra-Thor AGI frameworks",
            line,
        ):
            reason = "exact-case mismatch (Lattice, lattices, or Framework)"
        elif re.search(
            r"(?i)\bAGI\b\s*(?:vs\.?|versus|/)\s*\bASI\b|\bASI\b\s*(?:vs\.?|versus|/)\s*\bAGI\b",
            line,
        ):
            reason = "explanatory comparison such as AGI vs ASI"
        else:
            reason = "not one of the three exact epithet phrases"
            pre = text[max(0, pos - 48) : pos]
            wm = re.search(r"([A-Za-z0-9][A-Za-z0-9'’-]*)\s*$", pre)
            left_words[wm.group(1) if wm else "(start)"] += 1
        reasons[reason] += 1
        if sum(1 for r, _s in samples if r == reason) < 8:
            samples.append((reason, f"{rel}: {line.strip()[:200]}"))
    return reasons, samples, left_words


def suspicious(before: str, after: str) -> list[str]:
    flags = []
    if before == after:
        return flags
    b_lines = before.splitlines()
    a_lines = after.splitlines()
    if len(b_lines) != len(a_lines):
        flags.append("line count changed")
        return flags
    for n, (b, a) in enumerate(zip(b_lines, a_lines), 1):
        if b == a:
            continue
        if re.search(r"\b[Aa]n lattice\b", a):
            flags.append(f"{n}: article still unfit: {a[:220]}")
        if RX_DOUBLED.search(a) and (
            not RX_DOUBLED.search(b) or RX_DOUBLED.search(a).group(0) not in b
        ):
            flags.append(f"{n}: doubled word remains: {a[:220]}")
        if "  " in a and a.count("  ") > b.count("  "):
            flags.append(f"{n}: new double space: {a[:220]}")
        if RX_FRAMEWORK.search(a) or RX_RATHOR.search(a) or RX_LATTICE.search(a):
            # Still present only if it sits in a mask. Flag for the eye review.
            flags.append(f"{n}: epithet still on a changed line: {a[:220]}")
    return flags


def self_test() -> None:
    sample = (
        "See the Ra-Thor AGI framework, the sovereign Ra-Thor AGI lattice, "
        "an AGI lattice, QSA-AGI lattice, PATSAGi, QSA-AGi, NEXi-AGi, AGSi, "
        "AGi, `Ra-Thor AGI`, [Ra-Thor AGI](http://example.test/Ra-Thor%20AGI), "
        "and Ra-Thor AGI-guided work.\n"
        "```\ncode Ra-Thor AGI lattice\n```\n"
        "After the fence: Ra-Thor AGI.\n"
    )
    out, counts, _notes = apply_epithets(sample)
    expect = (
        "See the Ra-Thor framework, the sovereign Ra-Thor lattice, "
        "a lattice, QSA-AGI lattice, PATSAGi, QSA-AGi, NEXi-AGi, AGSi, "
        "AGi, `Ra-Thor AGI`, [Ra-Thor](http://example.test/Ra-Thor%20AGI), "
        "and Ra-Thor AGI-guided work.\n"
        "```\ncode Ra-Thor AGI lattice\n```\n"
        "After the fence: Ra-Thor.\n"
    )
    if out != expect:
        raise SystemExit(f"self-test mismatch:\n{out!r}\n!=\n{expect!r}")
    if counts["Ra-Thor AGI framework"] != 1:
        raise SystemExit(counts)
    # Sovereign lattice, link text, and the sentence after the fence.
    if counts["Ra-Thor AGI"] != 3:
        raise SystemExit(counts)
    if counts["AGI lattice"] != 1:
        raise SystemExit(counts)
    titled = "# Pillar 7 — " + OLD_TITLE + " TOLC-2026\n"
    titled_out, n = retitle(titled)
    if n != 1 or NEW_TITLE not in titled_out or OLD_TITLE in titled_out:
        raise SystemExit(titled_out)
    quoted = f'# Say "{OLD_TITLE}" here\n'
    quoted_out, nq = retitle(quoted)
    if nq != 0 or quoted_out != quoted:
        raise SystemExit(repr(quoted_out))
    prose = f"The {OLD_TITLE} is a sentence.\n"
    prose_out, np = retitle(prose)
    if np != 0 or prose_out != prose:
        raise SystemExit(prose_out)
    fenced = "```markdown\n# " + OLD_TITLE + "\nbody " + OLD_TITLE + "\n```\n"
    fenced_out, nf = retitle(fenced)
    if nf != 1 or fenced_out.count(NEW_TITLE) != 1 or fenced_out.count(OLD_TITLE) != 1:
        raise SystemExit(fenced_out)
    doubled = fix_grammar("lattice AGI lattice", "lattice lattice")
    if doubled != "lattice":
        raise SystemExit(doubled)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    self_test()

    token_before = {k: {"all": 0, "non_archive": 0} for k in PROTECTED_TOKENS}
    token_after = {k: {"all": 0, "non_archive": 0} for k in PROTECTED_TOKENS}
    mercy_before = mercy_after = 0
    pattern_counts: Counter = Counter()
    retitle_files: list[dict] = []
    banner_files: list[str] = []
    epithet_files: list[str] = []
    grammar_notes: list[str] = []
    suspicious_lines: list[str] = []
    remaining: Counter = Counter()
    remaining_samples: dict[str, list[str]] = {}
    free_left: Counter = Counter()
    files_considered = 0
    locked_skipped: list[str] = []
    tweet_skipped = False

    for path, rel in iter_md():
        files_considered += 1
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            print(f"skip undecodable {rel}", file=sys.stderr)
            continue
        archive = is_archive(rel)
        locked = filename_locked(rel) or content_locked(text)
        tweet = is_tweet(rel)
        if locked:
            locked_skipped.append(rel)
        if tweet:
            tweet_skipped = True

        for token in PROTECTED_TOKENS:
            c = literal_count(text, token)
            token_before[token]["all"] += c
            if not archive:
                token_before[token]["non_archive"] += c
        mercy_before += text.count(MERCY_SENTENCE)

        new = text
        epithet_changed = False
        if not archive and not locked and not tweet:
            new, counts, notes = apply_epithets(new)
            for key, n in counts.items():
                pattern_counts[key] += n
            if new != text:
                epithet_changed = True
                epithet_files.append(rel)
                grammar_notes.extend(f"{rel}: {note[:180]}" for note in notes)
                for flag in suspicious(text, new):
                    suspicious_lines.append(f"{rel}: {flag}")

        retitle_n = 0
        # Banner targets stay banner-only. Locked files and the tweet stay put.
        will_banner = archive and BANNER not in text and not locked
        if not locked and not tweet and not will_banner and OLD_TITLE in new:
            retitled, retitle_n = retitle(new)
            if retitle_n:
                new = retitled
                retitle_files.append({"file": rel, "count": retitle_n})

        bannered = False
        if will_banner:
            if new != text:
                raise SystemExit(f"banner target was otherwise edited: {rel}")
            new = BANNER + text
            bannered = True
            banner_files.append(rel)

        if not archive:
            reasons, samples, left_words = classify_agi(new, rel)
            for reason, n in reasons.items():
                remaining[reason] += n
            for reason, snip in samples:
                bucket = remaining_samples.setdefault(reason, [])
                if len(bucket) < 8:
                    bucket.append(snip)
            free_left.update(left_words)

        for token in PROTECTED_TOKENS:
            c = literal_count(new, token)
            token_after[token]["all"] += c
            if not archive:
                token_after[token]["non_archive"] += c
        mercy_after += new.count(MERCY_SENTENCE)

        if new != text and not args.dry_run:
            path.write_bytes(new.encode("utf-8"))

    # Second pass over written tree is unnecessary; counts are from new text.
    report = {
        "regexes": [
            {
                "pattern": "Ra-Thor AGI framework",
                "regex": RX_FRAMEWORK.pattern,
                "replacement": "Ra-Thor framework",
                "matches": pattern_counts["Ra-Thor AGI framework"],
            },
            {
                "pattern": "Ra-Thor AGI",
                "regex": RX_RATHOR.pattern,
                "replacement": "Ra-Thor",
                "matches": pattern_counts["Ra-Thor AGI"],
            },
            {
                "pattern": "AGI lattice",
                "regex": RX_LATTICE.pattern,
                "replacement": "lattice",
                "matches": pattern_counts["AGI lattice"],
            },
        ],
        "epithet_files": len(epithet_files),
        "retitle_files": retitle_files,
        "retitle_count": sum(item["count"] for item in retitle_files),
        "banner_files": banner_files,
        "banner_count": len(banner_files),
        "files_considered": files_considered,
        "locked_skipped": sorted(locked_skipped),
        "tweet_skipped": tweet_skipped,
        "token_before": token_before,
        "token_after": token_after,
        "mercy_before": mercy_before,
        "mercy_after": mercy_after,
        "remaining_agi": dict(remaining),
        "remaining_samples": remaining_samples,
        "free_agi_left_words": free_left.most_common(30),
        "grammar_notes": grammar_notes[:50],
        "suspicious": suspicious_lines[:80],
        "suspicious_count": len(suspicious_lines),
        "dry_run": args.dry_run,
    }
    text_out = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text_out)
    summary = {
        "patterns": report["regexes"],
        "epithet_files": report["epithet_files"],
        "retitle": report["retitle_files"],
        "banner_count": report["banner_count"],
        "banner_agsi_eval": sum("agsi-eval/" in p for p in banner_files),
        "mercy": [mercy_before, mercy_after],
        "tokens_non_archive": {
            k: [token_before[k]["non_archive"], token_after[k]["non_archive"]]
            for k in PROTECTED_TOKENS
        },
        "tokens_all_md": {
            k: [token_before[k]["all"], token_after[k]["all"]]
            for k in PROTECTED_TOKENS
        },
        "remaining_agi": report["remaining_agi"],
        "free_agi_left_words": report["free_agi_left_words"],
        "suspicious_count": len(suspicious_lines),
        "suspicious_head": suspicious_lines[:30],
        "grammar_head": grammar_notes[:20],
        "locked_skipped": report["locked_skipped"],
    }
    print(json.dumps(summary, indent=2))
    if mercy_before != mercy_after:
        print("MERCY SENTENCE COUNT CHANGED", file=sys.stderr)
        return 1
    for token in PROTECTED_TOKENS:
        if token == "AGi":
            continue
        if token_before[token] != token_after[token]:
            print(f"TOKEN CHANGED {token}", file=sys.stderr)
            return 1
    agi_delta_all = token_after["AGi"]["all"] - token_before["AGi"]["all"]
    if agi_delta_all != report["retitle_count"]:
        print(
            f"AGi delta {agi_delta_all} != retitles {report['retitle_count']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
