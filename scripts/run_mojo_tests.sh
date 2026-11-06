#!/usr/bin/env bash
# Run every tests/**/test_*.mojo file (or the ones given as arguments).
# Exits non-zero if any test file fails.
set -u
cd "$(dirname "$0")/.."
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  while IFS= read -r f; do
    files+=("$f")
  done < <(find tests -name "test_*.mojo" | sort)
fi
status=0
for f in "${files[@]}"; do
  echo "=== $f"
  out=$(mojo run -I . -I tests "$f" 2>&1)
  rc=$?
  printf '%s\n' "$out" | grep -v "Crashpad"
  if [ $rc -ne 0 ]; then status=1; echo "!!! $f FAILED (exit $rc)"; fi
done
exit $status
