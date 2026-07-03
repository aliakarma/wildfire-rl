"""Run-metadata / artifact-hashing tests (Phase 8)."""

from __future__ import annotations

import hashlib

from wildfire_rl.logging_utils import config_hash, file_sha256, run_metadata


def test_file_sha256(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert file_sha256(p) == hashlib.sha256(b"hello").hexdigest()
    assert file_sha256(None) is None
    assert file_sha256(tmp_path / "missing.bin") is None


def test_config_hash_stable():
    a = config_hash({"x": 1, "y": 2})
    b = config_hash({"y": 2, "x": 1})  # key order must not matter
    assert a == b and len(a) == 12


def test_run_metadata_records_artifacts(tmp_path):
    tensor = tmp_path / "state_tensor.npy"
    tensor.write_bytes(b"\x01\x02\x03")
    meta = run_metadata(
        config_dict={"env": {"grid_size": 32}}, seed=3, artifacts={"state_tensor": tensor}
    )
    assert meta["seed"] == 3
    assert "git_sha" in meta and "libraries" in meta
    assert "config_hash" in meta
    assert meta["artifacts"]["state_tensor"] == hashlib.sha256(b"\x01\x02\x03").hexdigest()
