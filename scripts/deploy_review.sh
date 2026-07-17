#!/usr/bin/env bash
# Review-channel deploy for the dashboard (Dashboard_Guide.md §15).
#
# Runs every release gate in order and only then deploys to the neutral Netlify site:
#   data build (fingerprint-gated, media copied) -> consistency --release -> unit tests
#   -> production build -> anonymity grep -> deploy.
#
# Runs locally (Git Bash or WSL) because the frozen checkpoints and the ~30 MB of MP4
# media live only on this machine — CI validates the committed data but cannot deploy.
#
# Usage:
#   bash scripts/deploy_review.sh            # dry run: all gates, no deploy
#   bash scripts/deploy_review.sh --deploy   # gates + `netlify deploy --prod`
#
# Deploy needs NETLIFY_AUTH_TOKEN and NETLIFY_SITE_ID in the environment (a neutral,
# non-identifying site name — see §15.1). Camera-ready: replace public/robots.txt with an
# allow-all file and de-anonymize About before deploying the accepted version.
set -euo pipefail
cd "$(dirname "$0")/.."

DEPLOY=0
[[ "${1:-}" == "--deploy" ]] && DEPLOY=1

echo "== 1/6 data build (fingerprint gate + media copy)"
python scripts/build_dashboard_data.py --copy-media

echo "== 2/6 consistency gate (--release: media + replays mandatory)"
python scripts/check_dashboard_consistency.py --release

echo "== 3/6 unit tests"
(cd dashboard && npm test)

echo "== 4/6 production build"
(cd dashboard && npm run build)

echo "== 5/6 anonymity grep over dist/"
if grep -rIiqE "akarma" dashboard/dist; then
  echo "ERROR: identifying strings found in dist/:" >&2
  grep -rIiE "akarma" dashboard/dist | head >&2
  exit 1
fi
echo "   dist/ clean"

echo "== 6/6 deploy"
if [[ "$DEPLOY" == "1" ]]; then
  : "${NETLIFY_AUTH_TOKEN:?set NETLIFY_AUTH_TOKEN}"
  : "${NETLIFY_SITE_ID:?set NETLIFY_SITE_ID}"
  (cd dashboard && npx --yes netlify-cli deploy --prod --dir dist --site "$NETLIFY_SITE_ID")
  echo "Review build deployed."
else
  echo "   dry run complete — re-run with --deploy to publish (needs NETLIFY_AUTH_TOKEN/NETLIFY_SITE_ID)"
fi
