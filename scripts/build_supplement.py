"""Build the anonymous AAAI Code & Data Supplement archive.

Stages an explicit **allowlist** of files into a clean tree, applies anonymization edits to
the staged copies (never in place), re-encodes the dashboard media down to the size cap,
emits checksums, hard-fails on any identity leak, and zips the result.

    python scripts/build_supplement.py
    python scripts/build_supplement.py --out supplement.zip --max-mb 48
    python scripts/build_supplement.py --skip-media     # fast iteration, no ffmpeg

This script is deliberately EXCLUDED from the archive it produces: it contains the literal
author strings used for leak scanning, which would defeat the purpose of shipping it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Identity strings. Any of these surviving into the staged tree fails the build.
# --------------------------------------------------------------------------------------
LEAK_PATTERNS = [
    r"aliakarma",
    r"Ali\s+Akarma",
    r"ALIAKA~1",
    r"vercel\.app",
    r"VERCEL_OIDC",
    r"aliakarmas-projects",
]
# Binary/opaque files we never scan (checksums make tampering evident instead).
LEAK_SCAN_SKIP_SUFFIXES = {
    ".pt", ".npy", ".mp4", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".nc",
    ".tif", ".gz", ".zip", ".woff", ".woff2", ".ico",
}

SEED42 = "_s42.pt"

# --------------------------------------------------------------------------------------
# Allowlist. (source, dest) plus a filter deciding which files come along.
# Paths are preserved from the repo wherever the code or the docs reference them by name,
# so every command quoted in RESULTS_FROZEN.md works verbatim inside the archive.
# --------------------------------------------------------------------------------------

def _no_cache(p: Path) -> bool:
    parts = set(p.parts)
    return not (parts & {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}) and p.suffix != ".pyc"


def py_only(p: Path) -> bool:
    return _no_cache(p) and p.suffix in {".py", ".typed", ".yaml", ".yml", ".cfg", ".toml"}


def everything(p: Path) -> bool:
    return _no_cache(p)


def results_seed42_ckpt(p: Path) -> bool:
    """Frozen results: all tables/curves/logs, plus only the seed-42 checkpoints."""
    if not _no_cache(p):
        return False
    if p.suffix == ".pt":
        return p.name.endswith(SEED42)
    return True


def results_no_ckpt(p: Path) -> bool:
    """Frozen results with no weights at all.

    Used for the ablation/transfer directories: re-running those studies loads checkpoints
    via ``--ckpt-dir wildfire_phase3_multiseed``, so their own copies are dead weight.
    """
    return _no_cache(p) and p.suffix != ".pt"


def simulator_source(p: Path) -> bool:
    """Cell2Fire C++ source, no build artifacts (the .gch alone is 194 MB)."""
    if not _no_cache(p):
        return False
    if p.suffix in {".gch", ".o", ".so", ".a", ".pyc"}:
        return False
    if p.name == "Cell2Fire" and p.suffix == "":  # compiled binary
        return False
    if "parallel_code" in p.parts:  # unused variant, ~0.1 MB but noise
        return False
    return True


def dashboard_source(p: Path) -> bool:
    """Dashboard sources only.

    ``public/media`` (548 MB of GIFs/PNGs) and ``dist`` are both excluded here: the prebuilt
    site is copied separately by ``copy_dist_shell`` and the media is re-encoded into
    ``dist/media`` by ``build_media``, so each ships exactly once.
    """
    if not _no_cache(p):
        return False
    blocked = {"node_modules", "test-results", "playwright-report", "dist", ".vercel"}
    if set(p.parts) & blocked:
        return False
    if p.parts[:2] == ("public", "media"):
        return False
    return p.name != ".env.local"


COPY_SPECS: list[tuple[str, str, object]] = [
    # --- code -------------------------------------------------------------------------
    ("src", "src", py_only),
    ("scripts", "scripts", py_only),
    ("configs", "configs", everything),
    ("tests", "tests", everything),
    # --- simulator (path must stay put: cell2fire_binding.DEFAULT_BINARY resolves it) ---
    ("third_party/firehose/cell2fire", "third_party/firehose/cell2fire", simulator_source),
    ("third_party/firehose/data", "third_party/firehose/data", everything),
    ("third_party/firehose/LICENSE", "third_party/firehose/LICENSE", everything),
    ("third_party/firehose/README.md", "third_party/firehose/README.md", everything),
    ("third_party/README.md", "third_party/README.md", everything),
    # --- data: simulator inputs + region tensors, not the 703 MB of raw rasters --------
    ("data/cell2fire", "data/cell2fire", everything),
    ("data/sample", "data/sample", everything),
    ("data/README.md", "data/README.md", everything),
    ("data/california/grids", "data/california/grids", everything),
    ("data/saudi_eastern_province/grids", "data/saudi_eastern_province/grids", everything),
    # --- frozen results (seed-42 checkpoints only) ------------------------------------
    ("wildfire_phase3_multiseed", "wildfire_phase3_multiseed", results_seed42_ckpt),
    ("wildfire_phase3_hiercomm_learned", "wildfire_phase3_hiercomm_learned", results_seed42_ckpt),
    ("wildfire_phase4", "wildfire_phase4", results_no_ckpt),
    ("wildfire_phase4_extended", "wildfire_phase4_extended", results_no_ckpt),
    ("wildfire_phase6", "wildfire_phase6", results_no_ckpt),
    # --- docs -------------------------------------------------------------------------
    ("docs/data_card.md", "docs/data_card.md", everything),
    ("docs/infra_card.md", "docs/infra_card.md", everything),
    ("docs/RESULTS_FROZEN.md", "docs/RESULTS_FROZEN.md", everything),
    ("docs/MIGRATION.md", "docs/MIGRATION.md", everything),
    # --- dashboard --------------------------------------------------------------------
    ("dashboard", "dashboard", dashboard_source),
    # --- root ---------------------------------------------------------------------------
    ("pyproject.toml", "pyproject.toml", everything),
    ("environment-linux.yml", "environment-linux.yml", everything),
    ("LICENSE", "LICENSE", everything),
    (".gitattributes", ".gitattributes", everything),
]

# Files inside otherwise-allowlisted trees that must never ship.
DENY_NAMES = {
    "build_supplement.py",   # contains the leak patterns above
    "deploy_review.sh",      # deployment plumbing
    ".env.local",
    ".env",
}

# Authored supplement assets: <staged path> <- supplement_assets/<name>
ASSET_FILES = {
    "README.md": "README.md",
    "docs/REPRODUCIBILITY.md": "REPRODUCIBILITY.md",
    "THIRD_PARTY_LICENSES.md": "THIRD_PARTY_LICENSES.md",
    "data/RAW_DATA_NOT_INCLUDED.md": "RAW_DATA_NOT_INCLUDED.md",
    "third_party/firehose/cell2fire/Cell2FireC/BUILD.md": "BUILD.md",
    "dashboard/README.md": "dashboard_README.md",
    "requirements.txt": "requirements.txt",
}


# --------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------
def stage(dest_root: Path) -> int:
    copied = 0
    for src_rel, dst_rel, keep in COPY_SPECS:
        src = REPO / src_rel
        dst = dest_root / dst_rel
        if not src.exists():
            print(f"  WARN missing source, skipped: {src_rel}")
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            continue
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            if path.name in DENY_NAMES or not keep(rel):
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied


def install_assets(dest_root: Path) -> None:
    assets = REPO / "supplement_assets"
    for staged_rel, asset_name in ASSET_FILES.items():
        source = assets / asset_name
        if not source.is_file():
            raise SystemExit(f"missing authored asset: {source}")
        target = dest_root / staged_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# --------------------------------------------------------------------------------------
# Anonymization (applied to staged copies only)
# --------------------------------------------------------------------------------------
def anonymize(dest_root: Path) -> list[str]:
    notes = []

    lic = dest_root / "LICENSE"
    if lic.is_file():
        text = lic.read_text(encoding="utf-8")
        text = re.sub(r"Copyright \(c\) (\d{4}) .*", r"Copyright (c) \1 The Authors", text)
        lic.write_text(text, encoding="utf-8")
        notes.append("LICENSE: copyright holder -> The Authors")

    pyproject = dest_root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        text = re.sub(r'^authors = .*$', 'authors = [{ name = "Anonymous Authors" }]',
                      text, flags=re.MULTILINE)
        text = re.sub(r'\n\[project\.urls\]\n(?:.*\n)*?(?=\n\[)', '\n', text)
        pyproject.write_text(text, encoding="utf-8")
        notes.append("pyproject.toml: authors anonymized, [project.urls] removed")

    # Absolute Windows provenance paths baked into the conversion reports.
    for js in dest_root.rglob("ignition_candidates.json"):
        blob = json.loads(js.read_text(encoding="utf-8"))
        src = blob.get("source")
        if isinstance(src, str) and ("\\" in src or ":" in src):
            marker = "data"
            idx = src.replace("\\", "/").find(f"/{marker}/")
            blob["source"] = src.replace("\\", "/")[idx + 1:] if idx >= 0 else f"{marker}/<region>/raw/firms/"
            js.write_text(json.dumps(blob, indent=2), encoding="utf-8")
            notes.append(f"{js.relative_to(dest_root)}: absolute source path -> relative")

    for report in dest_root.rglob("conversion_report.json"):
        text = report.read_text(encoding="utf-8")
        cleaned = re.sub(r'"[A-Za-z]:\\\\[^"]*?\\\\(data\\\\[^"]*)"', r'"\1"', text)
        if cleaned != text:
            report.write_text(cleaned, encoding="utf-8")
            notes.append(f"{report.relative_to(dest_root)}: absolute paths stripped")

    return notes


# --------------------------------------------------------------------------------------
# Media: re-encode MP4s, convert PNG posters to JPEG, rewrite media.json
# --------------------------------------------------------------------------------------
def build_media(dest_root: Path, crf: int, width: int) -> dict:
    from PIL import Image

    src_media = REPO / "dashboard" / "public" / "media"
    out_media = dest_root / "dashboard" / "dist" / "media"
    stats = {"mp4_in": 0, "mp4_out": 0, "png_in": 0, "jpg_out": 0, "n_mp4": 0, "n_poster": 0}
    poster_map: dict[str, str] = {}

    for mp4 in sorted(src_media.rglob("*.mp4")):
        rel = mp4.relative_to(src_media)
        dst = out_media / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", f"scale='min({width},iw)':-2:flags=lanczos",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
            check=True,
        )
        stats["mp4_in"] += mp4.stat().st_size
        stats["mp4_out"] += dst.stat().st_size
        stats["n_mp4"] += 1

    for png in sorted(src_media.rglob("*.png")):
        rel = png.relative_to(src_media)
        dst = (out_media / rel).with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(png) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width), Image.LANCZOS)
            im.save(dst, "JPEG", quality=82, optimize=True, progressive=True)
        stats["png_in"] += png.stat().st_size
        stats["jpg_out"] += dst.stat().st_size
        stats["n_poster"] += 1
        poster_map[f"media/{rel.as_posix()}"] = f"media/{rel.with_suffix('.jpg').as_posix()}"

    _rewrite_media_json(dest_root, out_media, poster_map)
    return stats


def _rewrite_media_json(dest_root: Path, out_media: Path, poster_map: dict[str, str]) -> None:
    """Point media.json at the re-encoded files: new byte counts, JPEG posters, no GIFs."""
    for data_dir in (
        dest_root / "dashboard" / "dist" / "data",
        dest_root / "dashboard" / "public" / "data",
    ):
        mj = data_dir / "media.json"
        if not mj.is_file():
            continue
        blob = json.loads(mj.read_text(encoding="utf-8"))
        for item in blob.get("items", []):
            item.pop("gif_fallback", None)  # source GIFs are excluded from the archive
            if item.get("poster") in poster_map:
                item["poster"] = poster_map[item["poster"]]
            src_path = out_media.parent / item["src"]
            if src_path.is_file():
                item["bytes"] = src_path.stat().st_size
        blob.setdefault("meta", {})["media_note"] = (
            "Videos re-encoded (H.264, capped width) and posters converted to JPEG to fit the "
            "50 MB supplement cap. Source GIFs are excluded; regenerate with "
            "scripts/render_phase5_gifs.py and scripts/render_phase6_transfer.py."
        )
        mj.write_text(json.dumps(blob, indent=1), encoding="utf-8")


def copy_dist_shell(dest_root: Path) -> None:
    """Prebuilt dashboard, minus media (media is re-encoded into it separately)."""
    src = REPO / "dashboard" / "dist"
    dst = dest_root / "dashboard" / "dist"
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] == "media":
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


# --------------------------------------------------------------------------------------
# Verification gates
# --------------------------------------------------------------------------------------
def leak_scan(dest_root: Path) -> list[str]:
    rx = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)
    hits = []
    for path in sorted(dest_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dest_root).as_posix()
        if rx.search(rel):
            hits.append(f"{rel}  (path)")
        if path.suffix.lower() in LEAK_SCAN_SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = rx.search(line)
            if m:
                hits.append(f"{rel}:{n}  matched {m.group(0)!r}")
                break
    return hits


def write_checksums(dest_root: Path) -> int:
    lines = []
    for path in sorted(dest_root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(dest_root).as_posix()}")
    (dest_root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def tree_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def report_breakdown(root: Path) -> None:
    rows = []
    for child in sorted(root.iterdir()):
        size = tree_size(child) if child.is_dir() else child.stat().st_size
        rows.append((child.name, size))
    for name, size in sorted(rows, key=lambda r: -r[1]):
        print(f"    {name:<38} {size / 1e6:8.2f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="supplement_AAAI2027.zip", help="output zip (repo-relative)")
    ap.add_argument("--stage", default="build/supplement", help="staging directory")
    ap.add_argument("--max-mb", type=float, default=48.0, help="hard size ceiling for the zip")
    ap.add_argument("--crf", type=int, default=28, help="x264 CRF for re-encoded video")
    ap.add_argument("--width", type=int, default=960, help="max video/poster width in px")
    ap.add_argument("--skip-media", action="store_true", help="skip ffmpeg/Pillow re-encoding")
    args = ap.parse_args()

    dest_root = (REPO / args.stage).resolve()
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    print("1/8  staging allowlisted files")
    n = stage(dest_root)
    print(f"     {n} files staged")

    print("2/8  installing authored supplement assets")
    install_assets(dest_root)

    print("3/8  copying prebuilt dashboard (without media)")
    copy_dist_shell(dest_root)

    print("4/8  media")
    if args.skip_media:
        print("     skipped (--skip-media)")
    else:
        s = build_media(dest_root, args.crf, args.width)
        print(f"     {s['n_mp4']} videos {s['mp4_in']/1e6:.1f} -> {s['mp4_out']/1e6:.1f} MB")
        print(f"     {s['n_poster']} posters {s['png_in']/1e6:.1f} -> {s['jpg_out']/1e6:.1f} MB")

    print("5/8  anonymizing staged copies")
    for note in anonymize(dest_root):
        print(f"     {note}")

    print("6/8  leak scan")
    hits = leak_scan(dest_root)
    if hits:
        print(f"     {len(hits)} LEAK(S) FOUND — build aborted:")
        for h in hits[:40]:
            print(f"       {h}")
        return 1
    print("     clean — no identity strings in the staged tree")

    print("7/8  checksums")
    print(f"     SHA256SUMS.txt over {write_checksums(dest_root)} files")

    staged = tree_size(dest_root)
    print(f"\n     staged tree: {staged / 1e6:.2f} MB")
    report_breakdown(dest_root)

    print("\n8/8  zipping")
    out_zip = (REPO / args.out).resolve()
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(dest_root.rglob("*")):
            if path.is_file():
                zf.write(path, Path("supplement") / path.relative_to(dest_root))

    size_mb = out_zip.stat().st_size / 1e6
    print(f"     {out_zip.name}: {size_mb:.2f} MB")
    if size_mb > args.max_mb:
        print(f"\nFAILED: {size_mb:.2f} MB exceeds the {args.max_mb} MB ceiling.")
        return 1
    print(f"\nOK — {out_zip}")
    print(f"   {size_mb:.1f} MB of the 50 MB limit ({50 - size_mb:.1f} MB headroom)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
