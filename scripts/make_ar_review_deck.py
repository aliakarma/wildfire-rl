"""Generate the native-Arabic review deck for the dashboard copy (release gate §10.3).

Reads both locale resource files and writes ``docs/ar_copy_review.md`` — a side-by-side
EN/AR table a native Arabic reviewer with ML familiarity can work through, with review
instructions and a sign-off checklist. Regenerate whenever the locales change; the
reviewer's signed-off copy is the release gate the tests cannot automate (the vitest
suite guarantees structural parity only, not phrasing quality).

    python scripts/make_ar_review_deck.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCALES = REPO / "dashboard" / "src" / "i18n" / "locales"
OUT = REPO / "docs" / "ar_copy_review.md"

HEADER = f"""# Arabic copy review deck — Wildfire-MARL dashboard

*Generated {date.today()} from `dashboard/src/i18n/locales/*.json` by
`scripts/make_ar_review_deck.py`. Regenerate after any copy change.*

## Instructions for the reviewer

You are reviewing the **Arabic locale of a public dashboard accompanying a peer-reviewed
AAAI submission**. Please check every row for:

1. **Faithfulness** — the Arabic must not make a claim stronger or weaker than the
   English (e.g., "best on average, statistically tied on Saudi" must survive exactly).
2. **Terminology** — statistical and ML terms should read naturally to an Arabic-speaking
   ML researcher; method names (HierComm, CommNet, MAPPO) stay in Latin script.
3. **Register** — Modern Standard Arabic, scientific tone, no colloquialisms.
4. **Placeholders** — `{{{{x}}}}` markers and `<bdi>` tags must stay exactly where they are
   (they inject live numbers and protect them from right-to-left reordering).

Mark corrections directly in this file (or in `ar.json`), then complete the sign-off
block at the end.

"""

SIGNOFF = """
## Sign-off (release gate — Dashboard_Guide.md §10.3/§16)

- [ ] Every string reviewed against its English source
- [ ] No claim is stronger or weaker in Arabic than in English
- [ ] Statistical terminology verified by a reviewer with ML familiarity
- [ ] Method names remain in Latin script; hashes/seeds/paths render LTR
- [ ] Corrections applied to `dashboard/src/i18n/locales/ar.json` and `npm test` passes

Reviewer name: ______________________  Date: ____________

"""


def leaves(node: dict, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, str):
            out.append((path, value))
        else:
            out.extend(leaves(value, path))
    return out


def esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    en = dict(leaves(json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))))
    ar = dict(leaves(json.loads((LOCALES / "ar.json").read_text(encoding="utf-8"))))
    assert set(en) == set(ar), "locale key sets diverge — run `npm test` in dashboard/"

    sections: dict[str, list[str]] = {}
    for key in en:
        sections.setdefault(key.split(".")[0], []).append(key)

    lines = [HEADER]
    for section in sorted(sections):
        lines.append(f"## `{section}.*`\n")
        lines.append("| Key | English | Arabic | OK? |")
        lines.append("|---|---|---|---|")
        for key in sorted(sections[section]):
            lines.append(f'| `{key}` | {esc(en[key])} | <div dir="rtl">{esc(ar[key])}</div> |  |')
        lines.append("")
    lines.append(SIGNOFF)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(en)} strings)")


if __name__ == "__main__":
    main()
