#!/usr/bin/env bash
# Guard against R8 renaming a reflectively-loaded class (such as the WebView
# @JavascriptInterface bridge or Activity entry point) in release builds.
#
# Usage: scripts/check-r8-kept-classes.sh [path/to/mapping.txt]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ge 1 ]; then
  MAP="$1"
else
  MAP=""
  for cand in \
    android/app/build/outputs/mapping/release/mapping.txt \
    android/app/build/intermediates/mapping/release/minifyReleaseWithR8/mapping.txt; do
    if [ -f "$cand" ]; then
      if [ -z "$MAP" ] || [ "$cand" -nt "$MAP" ]; then MAP="$cand"; fi
    fi
  done
  if [ -z "$MAP" ]; then
    MAP=$(find android/app/build -name 'mapping.txt' -path '*release*' -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2- || true)
  fi
  [ -n "$MAP" ] || MAP="android/app/build/outputs/mapping/release/mapping.txt"
fi
LIST="scripts/r8-must-keep-classes.txt"

if [ ! -f "$MAP" ]; then
  echo "✖ mapping.txt not found (searched known paths + android/app/build)." >&2
  echo "  Run (cd android && ./gradlew :app:assembleRelease) first." >&2
  exit 2
fi

fqns=()
while IFS= read -r line; do
  [ -n "$line" ] && fqns+=("$line")
done < <(grep -vE '^\s*(#|$)' "$LIST" || true)

mapfile -t fqns < <(printf '%s\n' "${fqns[@]}" | sort -u)

failures=()
for fqn in "${fqns[@]}"; do
  [ -n "$fqn" ] || continue
  esc=${fqn//./\\.}
  esc=${esc//\$/\\\$}
  if grep -qE "^${esc} -> ${esc}:" "$MAP"; then
    continue
  fi
  actual=$(grep -E "^${esc} -> " "$MAP" | head -1 || true)
  if [ -n "$actual" ]; then
    failures+=("$fqn  →  RENAMED to ${actual##*-> }")
  else
    failures+=("$fqn  →  ABSENT (shrunk away or never compiled in)")
  fi
done

if [ ${#failures[@]} -gt 0 ]; then
  echo "✖ ${#failures[@]} reflectively-loaded class(es) did NOT survive R8 intact:"
  printf '   %s\n' "${failures[@]}"
  echo
  echo "Add or fix a -keep rule in android/app/proguard-rules.pro, then re-run."
  exit 1
fi

echo "✓ All ${#fqns[@]} reflectively-loaded classes kept their names through R8."
