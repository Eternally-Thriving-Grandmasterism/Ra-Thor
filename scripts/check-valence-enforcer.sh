#!/usr/bin/env bash
# Run the fenca audit that the valence-enforcer job gates on.
#
# Exit 0 only when the fenca tests pass and every corpus file audits as
# passed with audit_score 1.0. Missing inputs, cargo errors, a non-quarter
# score, and unparseable output exit non-zero. No valence constant is printed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${FENCA_MANIFEST:-$ROOT/crates/fenca/Cargo.toml}"
CORPUS="${FENCA_CORPUS:-$ROOT/scripts/valence-enforcer-corpus.txt}"

export CARGO_TERM_COLOR=never
export PYTHONUNBUFFERED=1

parse_audit() {
  local status="$1"
  local expected_json="$2"
  local audit_json="$3"
  AUDIT_JSON="$audit_json" python3 - "$status" "$expected_json" <<'PY'
import json
import os
import sys

status = int(sys.argv[1])
expected = json.loads(sys.argv[2])
raw = os.environ.get("AUDIT_JSON", "")
try:
    reports = json.loads(raw)
except json.JSONDecodeError as exc:
    print(f"parse failure: {exc}", file=sys.stderr)
    sys.exit(1)
if not isinstance(reports, list) or not reports:
    print("parse failure: expected a non-empty JSON list", file=sys.stderr)
    sys.exit(1)
if not isinstance(expected, list) or len(reports) != len(expected):
    print(
        f"parse failure: expected {len(expected) if isinstance(expected, list) else 'a list'} reports, got {len(reports)}",
        file=sys.stderr,
    )
    sys.exit(1)

quarters = (0.0, 0.25, 0.5, 0.75, 1.0)
required_checks = [
    "non_empty",
    "within_size_cap",
    "sources_when_required",
    "no_banned_overclaim",
]
all_passed = True
for report, path in zip(reports, expected):
    if not isinstance(report, dict):
        print("parse failure: report is not an object", file=sys.stderr)
        sys.exit(1)
    for key in ("path", "passed", "audit_score", "checks_performed", "reasons"):
        if key not in report:
            print(f"parse failure: missing {key}", file=sys.stderr)
            sys.exit(1)
    if report["path"] != path:
        print(f"parse failure: path {report['path']!r} != {path!r}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(report["passed"], bool):
        print("parse failure: passed is not a boolean", file=sys.stderr)
        sys.exit(1)
    score = report["audit_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        print("parse failure: audit_score is not a number", file=sys.stderr)
        sys.exit(1)
    if not any(abs(float(score) - quarter) < 1e-9 for quarter in quarters):
        print(
            f"parse failure: audit_score {score} is not a fraction of the four checks",
            file=sys.stderr,
        )
        sys.exit(1)
    if report["checks_performed"] != required_checks:
        print(f"parse failure: unexpected checks {report['checks_performed']}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(report["reasons"], list):
        print("parse failure: reasons is not a list", file=sys.stderr)
        sys.exit(1)
    print(
        f"measured {path} passed={str(report['passed']).lower()} "
        f"audit_score={float(score):.2f} reasons={report['reasons']}"
    )
    if report["passed"] is not True or abs(float(score) - 1.0) > 1e-9 or report["reasons"]:
        all_passed = False

if status == 0 and not all_passed:
    print("tool error: exit 0 but an audit did not pass", file=sys.stderr)
    sys.exit(1)
if status == 1 and all_passed:
    print("tool error: exit 1 but every audit passed", file=sys.stderr)
    sys.exit(1)
if not all_passed:
    print("fenca audit failed", file=sys.stderr)
    sys.exit(1)
if status != 0:
    print(f"tool error: fenca-audit exited {status}", file=sys.stderr)
    sys.exit(1)
print(f"fenca audit passed for {len(reports)} files")
PY
}

expect_fail() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "parser self-check failed open: $label" >&2
    exit 1
  fi
}

parser_self_check() {
  local good='[{"path":"/self-check","passed":true,"audit_score":1.0,"checks_performed":["non_empty","within_size_cap","sources_when_required","no_banned_overclaim"],"reasons":[]}]'
  local bad='[{"path":"/self-check","passed":false,"audit_score":0.75,"checks_performed":["non_empty","within_size_cap","sources_when_required","no_banned_overclaim"],"reasons":["rejected: banned overclaim marker '\''perfect valence'\''"]}]'
  local fake='[{"path":"/self-check","passed":true,"audit_score":0.9999997,"checks_performed":["non_empty","within_size_cap","sources_when_required","no_banned_overclaim"],"reasons":[]}]'
  local expect='["/self-check"]'

  expect_fail "invalid json" parse_audit 0 "$expect" "not-json"
  expect_fail "fabricated constant" parse_audit 0 "$expect" "$fake"
  expect_fail "banned marker" parse_audit 1 "$expect" "$bad"
  parse_audit 0 "$expect" "$good" >/dev/null
  echo "parser self-check passed"
}

if [[ "${1:-}" == "parser-self-check" ]]; then
  parser_self_check
  exit 0
fi

parser_self_check

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing input: fenca manifest not found at $MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$CORPUS" ]]; then
  echo "missing input: corpus not found at $CORPUS" >&2
  exit 1
fi

workflow="$ROOT/.github/workflows/valence-enforcer.yml"
if [[ ! -f "$workflow" ]]; then
  echo "missing input: $workflow" >&2
  exit 1
fi
if grep -Eq 'VALENCE=0\.999999|valence=0\.999999' "$workflow"; then
  echo "workflow still writes a constant valence" >&2
  exit 1
fi

rels=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  if [[ "$line" = /* || "$line" == *..* ]]; then
    echo "missing input: corpus path must stay inside the repo: $line" >&2
    exit 1
  fi
  if [[ ! -f "$ROOT/$line" ]]; then
    echo "missing input: $line" >&2
    exit 1
  fi
  rels+=("$line")
done < "$CORPUS"

if [[ ${#rels[@]} -eq 0 ]]; then
  echo "missing input: corpus is empty" >&2
  exit 1
fi

echo "running fenca tests"
set +e
test_log="$(cargo test --manifest-path "$MANIFEST" --locked -- --test-threads=1 2>&1)"
test_status=$?
set -e
printf '%s\n' "$test_log"
if [[ "$test_status" -ne 0 ]]; then
  echo "tool error: cargo test exited $test_status" >&2
  exit 1
fi
if [[ "$test_log" != *"test known_bad_input_fails_with_reasons ... ok"* ]]; then
  echo "parse failure: known-bad fenca test did not pass" >&2
  exit 1
fi
if [[ "$test_log" != *"test known_good_input_passes ... ok"* ]]; then
  echo "parse failure: known-good fenca test did not pass" >&2
  exit 1
fi
if [[ "$test_log" != *"test banned_marker_exits_one_with_three_quarter_score ... ok"* ]]; then
  echo "parse failure: audit binary fail-path test did not pass" >&2
  exit 1
fi

manifest_dir="$(cd "$(dirname "$MANIFEST")" && pwd)"
set +e
build_log="$(cargo build --manifest-path "$MANIFEST" --locked --bin fenca_audit --quiet 2>&1)"
build_status=$?
set -e
if [[ "$build_status" -ne 0 ]]; then
  printf '%s\n' "$build_log" >&2
  echo "tool error: cargo build exited $build_status" >&2
  exit 1
fi

bin="$manifest_dir/target/debug/fenca_audit"
if [[ ! -x "$bin" ]]; then
  echo "tool error: fenca_audit binary missing at $bin" >&2
  exit 1
fi

abs=()
expected_json='['
for rel in "${rels[@]}"; do
  abs+=("$ROOT/$rel")
  if [[ "$expected_json" != '[' ]]; then
    expected_json+=','
  fi
  expected_json+="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$ROOT/$rel")"
done
expected_json+=']'

set +e
audit_json="$("$bin" "${abs[@]}")"
audit_status=$?
set -e
if [[ "$audit_status" -ne 0 && "$audit_status" -ne 1 ]]; then
  echo "tool error: fenca_audit exited $audit_status" >&2
  printf '%s\n' "$audit_json" >&2
  exit 1
fi

parse_audit "$audit_status" "$expected_json" "$audit_json"
