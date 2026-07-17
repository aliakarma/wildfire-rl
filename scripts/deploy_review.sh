#!/usr/bin/env bash
# Review-channel deploy for the dashboard (Dashboard_Guide.md §15) — Vercel, MP4-only.
#
# Runs every release gate in order, prunes the source GIFs from the build (the dashboard
# plays MP4 + PNG posters — MediaFrame never renders the ~455 MB of GIFs), and only then
# deploys the prebuilt static dist/ to Vercel:
#   data build (fingerprint-gated, media copied) -> consistency --release -> unit tests
#   -> production build -> MP4-only prune -> anonymity grep -> deploy.
#
# Runs locally (Git Bash or WSL) because the frozen checkpoints and the media live only on
# this machine — CI validates the committed data but cannot deploy. The pruned dist/ is
# ~80 MB (MP4 + posters) instead of ~550 MB.
#
# Usage:
#   bash scripts/deploy_review.sh            # dry run: all gates + prune, no deploy
#   bash scripts/deploy_review.sh --deploy   # gates + prune + `vercel deploy --prod`
#
# One-time setup (creates a stable, neutrally named project; link lives in
# dashboard/.vercel and survives rebuilds because only dist/ is regenerated):
#   npm i -g vercel
#   (cd dashboard && vercel link)            # pick a neutral, non-identifying project name
#
# Deploy needs VERCEL_TOKEN in the environment (create at
# https://vercel.com/account/tokens). Optionally set VERCEL_SCOPE to a team/org slug if the
# project lives under a team rather than your personal account. Double-blind review (§15.1):
# keep the project name and About page anonymous and public/robots.txt at "Disallow: /".
# Camera-ready: replace robots.txt with an allow-all file and de-anonymize About first.
set -euo pipefail
cd "$(dirname "$0")/.."

DEPLOY=0
[[ "${1:-}" == "--deploy" ]] && DEPLOY=1

echo "== 1/7 data build (fingerprint gate + media copy)"
python scripts/build_dashboard_data.py --copy-media

echo "== 2/7 consistency gate (--release: media + replays mandatory)"
python scripts/check_dashboard_consistency.py --release

echo "== 3/7 unit tests"
(cd dashboard && npm test)

echo "== 4/7 production build"
(cd dashboard && npm run build)

echo "== 5/7 MP4-only prune (drop source GIFs from dist/)"
# gif_fallback in media.json is intentionally left dangling: nothing loads it (MediaFrame
# uses src/poster only) and check_dashboard_consistency --release verifies src + poster,
# not gif_fallback — so the pruned build still passes every gate.
before=$(du -sh dashboard/dist | cut -f1)
find dashboard/dist/media -name '*.gif' -delete 2>/dev/null || true
after=$(du -sh dashboard/dist | cut -f1)
echo "   dist/ pruned: ${before} -> ${after} (dashboard plays MP4 + poster only)"

echo "== 6/7 anonymity grep over dist/"
if grep -rIiqE "akarma" dashboard/dist; then
  echo "ERROR: identifying strings found in dist/:" >&2
  grep -rIiE "akarma" dashboard/dist | head >&2
  exit 1
fi
echo "   dist/ clean"

echo "== 7/7 deploy to Vercel"
if [[ "$DEPLOY" == "1" ]]; then
  : "${VERCEL_TOKEN:?set VERCEL_TOKEN — create one at https://vercel.com/account/tokens}"
  scope=()
  [[ -n "${VERCEL_SCOPE:-}" ]] && scope=(--scope "$VERCEL_SCOPE")
  # Deploy the pruned static dist/ to the project linked in dashboard/.vercel. Passing the
  # explicit `dist` path makes Vercel upload it as a static site (no package.json inside →
  # no build step), so the gitignored media ships without a git-based build.
  (cd dashboard && npx --yes vercel deploy dist --prod --yes \
      --token "$VERCEL_TOKEN" "${scope[@]+"${scope[@]}"}")
  echo "Deployed to Vercel (production)."
else
  echo "   dry run complete — re-run with --deploy to publish"
  echo "   (needs VERCEL_TOKEN + a one-time 'cd dashboard && vercel link')"
fi
