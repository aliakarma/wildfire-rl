"""Compress the rollout GIFs to MP4 (H.264) for the dashboard media gallery.

The pre-rendered rollout GIFs run 5–69 MB each (~460 MB total) — the single biggest
dashboard UX cost. H.264 at the same resolution/frame timing is typically 10–20× smaller
with no visual loss for these flat-color frames. GIFs stay on disk untouched (they remain
the archival artifact); each ``<name>.gif`` gains a ``<name>.mp4`` sibling, which
``build_dashboard_data.py`` then prefers when building the media index.

Uses the static ffmpeg binary bundled with ``imageio-ffmpeg`` (pure-wheel install, no
system ffmpeg needed)::

    pip install imageio-ffmpeg
    python scripts/compress_media.py            # convert missing/stale MP4s
    python scripts/compress_media.py --force    # reconvert everything
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

REPO = Path(__file__).resolve().parents[1]
GIF_DIRS = [REPO / "figures" / "wildfire_phase5_gifs", REPO / "figures" / "wildfire_phase6_gifs"]


def convert(ffmpeg: str, gif: Path, force: bool) -> str:
    mp4 = gif.with_suffix(".mp4")
    if mp4.exists() and not force and mp4.stat().st_mtime >= gif.stat().st_mtime:
        return "kept"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(gif),
        # yuv420p needs even dimensions; GIF frame timing is preserved by ffmpeg.
        "-vf",
        "crop=floor(iw/2)*2:floor(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)
    return "wrote"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="reconvert even if MP4 is current")
    args = ap.parse_args()

    ffmpeg = get_ffmpeg_exe()
    total_gif = total_mp4 = 0
    for d in GIF_DIRS:
        for gif in sorted(d.glob("*.gif")):
            status = convert(ffmpeg, gif, args.force)
            mp4 = gif.with_suffix(".mp4")
            g, m = gif.stat().st_size, mp4.stat().st_size
            total_gif += g
            total_mp4 += m
            print(
                f"  {status} {mp4.name}: {g / 1e6:.1f} MB -> {m / 1e6:.1f} MB "
                f"({g / max(m, 1):.0f}x)",
                flush=True,
            )
    print(f"\nTotal: {total_gif / 1e6:.0f} MB GIF -> {total_mp4 / 1e6:.0f} MB MP4")


if __name__ == "__main__":
    main()
