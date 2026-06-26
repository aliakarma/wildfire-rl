"""CLI smoke tests (no training / no heavy deps)."""

from __future__ import annotations

import pytest

from wildfire_rl.cli import main


def test_info_runs(capsys):
    rc = main(["info", "--set", "seed=3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"version"' in out
    assert '"seed": 3' in out


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
