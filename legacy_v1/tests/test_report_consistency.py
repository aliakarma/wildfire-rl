"""Report ↔ CSV consistency (Phase 13).

Guards against two regressions:
  1. The untraceable/fabricated literals purged in Phase 12 must not resurface in the report.
  2. Every table embedded in ``docs/paper/report.md`` must equal what ``build_report_tables.py``
     renders from the current committed CSVs — so report numbers cannot silently drift from data.

Both degrade gracefully (skip) when the CSVs are absent, so a bare checkout does not error.
"""

from __future__ import annotations

import importlib.util
import pathlib

REPORT = pathlib.Path("docs/paper/report.md")
PURGED_LITERALS = ["-14581", "1564.89", "-38672.41", "1306.86", "Cohen's d > 11"]


def _report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _load_generator():
    spec = importlib.util.spec_from_file_location("brt", "scripts/build_report_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_untraceable_literals():
    """The fabricated numbers removed in Phase 12 must stay gone."""
    txt = _report_text()
    for bad in PURGED_LITERALS:
        assert bad not in txt, f"untraceable/fabricated literal resurfaced in report.md: {bad!r}"


def test_report_tables_match_csvs():
    """Every rendered data row (from the current CSVs) must appear verbatim in the report."""
    brt = _load_generator()
    report = _report_text()
    checked = 0
    for csv_name, title, cols, prov in brt.TABLES:
        block = brt.render_table(csv_name, title, cols, prov)
        if block is None:  # CSV absent in this checkout -> skip
            continue
        for line in block.splitlines():
            # data rows only: markdown table rows that aren't the header sep or the header itself
            if not line.startswith("| ") or set(line) <= {"|", "-", " "}:
                continue
            if line.lstrip("| ").split(" |")[0] in cols[:1]:  # header row (first col name)
                continue
            assert line in report, (
                f"report.md is out of sync with results/{csv_name}: missing row\n  {line}\n"
                f"Regenerate: python scripts/build_report_tables.py --out docs/paper/_generated_tables.md "
                f"and update report §16."
            )
            checked += 1
    assert checked > 0, "no CSV-derived rows were checked (are the certified CSVs present?)"
