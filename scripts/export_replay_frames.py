"""Export per-step replay frames from the frozen checkpoints (Dashboard_Guide.md §11.2).

Reproduces the evaluation rollout **exactly** (``phase3_eval.rollout_episode``: fresh env,
``reset(seed = group + EVAL_OFFSET + episode)``, ``apply_target_compliance = False``,
``select_actions`` each step) while capturing per-step state, and writes one JSON per
rollout into ``dashboard/public/data/replays/`` plus an ``index.json``.

**Self-verifying:** each native replay's final WEL/ISR/burned must equal the matching
``results/wildfire_phase3_multiseed/phase3_raw.csv`` row, and each cross-region replay the matching
``results/wildfire_phase6/transfer_matrix_raw.csv`` row — otherwise the export aborts. Replays are
therefore provably the rollouts behind the reported numbers.

Frame encoding: sparse cell-index deltas (``y*W + x``) instead of the guide's illustrative
RLE — equally compact for 32×32 deltas and trivially decodable in the browser.

Runs in the WSL venv (needs torch + Cell2Fire), like all result tooling::

    python scripts/export_replay_frames.py                 # 7 policies × 2 regions + 4 transfer
    python scripts/export_replay_frames.py --policies hiercomm_heur --regions saudi --no-transfer
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.eval.metrics import (  # noqa: E402
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import EVAL_OFFSET, load_nets, select_actions  # noqa: E402

ALL_POLICIES = [
    "noop",
    "value_first",
    "greedy_risk",
    "local_reactive",
    "mappo",
    "commnet",
    "hiercomm_heur",
]
LEARNED = {"mappo", "commnet", "hiercomm_heur", "hiercomm", "qmix"}
HIERARCHICAL = {"hiercomm_heur", "hiercomm"}
REGIONS = ["saudi", "california"]
TRANSFER_POLICIES = ["hiercomm_heur", "commnet"]

PHASE3_RAW = REPO / "results" / "wildfire_phase3_multiseed" / "phase3_raw.csv"
TRANSFER_RAW = REPO / "results" / "wildfire_phase6" / "transfer_matrix_raw.csv"
OUT_DEFAULT = REPO / "dashboard" / "public" / "data" / "replays"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cells(mask: np.ndarray) -> list[int]:
    """Flatten a boolean grid to sorted 0-indexed cell ids (y * W + x)."""
    ys, xs = np.nonzero(mask)
    w = mask.shape[1]
    return sorted(int(y) * w + int(x) for y, x in zip(ys, xs, strict=False))


def capture_rollout(
    region: str, kind: str, ckpt_region: str, seed: int, episode: int, ckpt_dir: str, device
) -> dict:
    """Run one eval-stream episode, capturing per-step frames.

    Control flow is byte-for-byte the evaluation procedure; only read-only state
    captures are added between steps.
    """
    set_seed(seed)
    env = make_marl_env(region, regime="default")
    train_seed = seed if kind in LEARNED else None
    nets = load_nets(kind, ckpt_region, ckpt_dir, env.num_agents, device, seed=train_seed)

    episode_seed = seed + EVAL_OFFSET + episode
    obs, info = env.reset(seed=episode_seed)
    env.apply_target_compliance = False

    inner = env.env
    h, w = inner.fire_state.shape
    crit = inner.criticality
    crit_max = float(crit.max()) if crit is not None and crit.max() > 0 else 1.0
    asset_cells = [
        {
            "y": int(y),
            "x": int(x),
            "type": int(inner.asset_type[y, x]),
            "value": float(inner.asset_values[int(inner.asset_type[y, x])]),
        }
        for y, x in zip(*np.nonzero(inner.asset_type > 0), strict=False)
    ]
    static = {
        "fuel": _cells(inner.fuel_mask > 0),
        "criticality": [
            round(float(v), 4)
            for v in (crit / crit_max if crit is not None else np.zeros((h, w))).ravel()
        ],
        "assets": asset_cells,
    }

    def frame(t: int, prev_fire: np.ndarray, prev_treated: np.ndarray) -> dict:
        fire = inner.fire_state > 0
        treated = inner.fire_state < 0
        agents = [[int(y), int(x)] for y, x in (env.agent_positions[a] for a in env.agents)]
        targets = (
            [[int(y), int(x)] for y, x in (env.strategic_targets[a] for a in env.agents)]
            if kind in HIERARCHICAL
            else []
        )
        return (
            {
                "t": t,
                "fire": _cells(fire & ~prev_fire),
                "treated": _cells(treated & ~prev_treated),
                "agents": agents,
                "targets": targets,
                "wel": round(
                    float(
                        weighted_economic_loss(
                            inner.asset_type, inner.fire_state, inner.asset_values
                        )
                    ),
                    6,
                ),
                "isr": round(
                    float(infrastructure_survival_rate(inner.asset_type, inner.fire_state)), 6
                ),
                "burned": int((inner.fire_state > 0).sum()),
            },
            fire,
            treated,
        )

    frames: list[dict] = []
    prev_fire = np.zeros((h, w), dtype=bool)
    prev_treated = np.zeros((h, w), dtype=bool)
    f0, prev_fire, prev_treated = frame(0, prev_fire, prev_treated)
    frames.append(f0)

    done, sc, ep_ret, ce = False, 0, 0.0, 1.0
    while not done:
        acts = select_actions(env, kind, nets, obs, info, device, sc)
        obs, rew, term, trunc, info = env.step(acts)
        ep_ret += float(rew["agent_0"])
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        sc += 1
        f, prev_fire, prev_treated = frame(sc, prev_fire, prev_treated)
        frames.append(f)

    final = {
        "WEL": float(
            weighted_economic_loss(inner.asset_type, inner.fire_state, inner.asset_values)
        ),
        "ISR": float(infrastructure_survival_rate(inner.asset_type, inner.fire_state)),
        "CE": float(ce),
        "burned": int((inner.fire_state > 0).sum()),
        "return": ep_ret,
    }
    env.close()

    return {
        "meta": {
            "region": region,
            "policy": kind,
            "ckpt_region": ckpt_region,
            "train_seed": seed if kind in LEARNED else None,
            "episode": episode,
            "episode_seed": episode_seed,
            "grid": [h, w],
            "steps": sc,
            # Full precision — rounding happens only at render time (data-layer rule §4.2),
            # and verification against the frozen CSVs is exact (abs_tol 1e-9).
            "final": final,
        },
        "static": static,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Self-verification against the frozen raw CSVs
# ---------------------------------------------------------------------------


class VerificationError(RuntimeError):
    """The captured rollout does not match the frozen per-episode record."""


def _verify(label: str, final: dict, row: dict[str, str]) -> None:
    """A replay is exported only if WEL, ISR, CE, and burned all match the frozen record
    to 1e-9 — four independent trajectory-derived statistics (CE and burned depend on the
    whole trajectory; terminal WEL/ISR alone can saturate identically under diverged
    trajectories).

    The frozen ``return`` column is not compared directly: it records the outcome
    objective ``-(WEL + 20·(1-ISR) + 0.1·collision_events)``, and the collision count is
    not persisted anywhere recomputable — its coordination component is already covered
    by the exact CE match.
    """
    for key, col in (("WEL", "WEL"), ("ISR", "ISR"), ("CE", "CE")):
        got, want = float(final[key]), float(row[col])
        if not math.isclose(got, want, rel_tol=0, abs_tol=1e-9):
            raise VerificationError(
                f"{label}: replay {key} {got!r} != frozen {want!r} — "
                f"the replay is NOT the reported rollout."
            )
    if int(final["burned"]) != int(float(row["burned"])):
        raise VerificationError(f"{label}: burned {final['burned']} != {row['burned']}")


def verify_native(final: dict, region: str, kind: str, seed: int, episode: int) -> None:
    with open(PHASE3_RAW, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["region"] == region
                and row["policy"] == kind
                and int(row["seed"]) == seed
                and int(row["episode"]) == episode
                and (kind not in LEARNED or float(row["train_seed"]) == seed)
            ):
                _verify(f"{region}/{kind}", final, row)
                return
    raise VerificationError(f"No phase3_raw.csv row for {region}/{kind}/seed{seed}/ep{episode}")


def verify_transfer(
    final: dict, eval_region: str, ckpt_region: str, kind: str, seed: int, episode: int
) -> None:
    with open(TRANSFER_RAW, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["policy"] == kind
                and row["ckpt_region"] == ckpt_region
                and row["eval_region"] == eval_region
                and int(row["eval_seed"]) == seed
                and int(float(row["train_seed"])) == seed
                and int(row["episode"]) == episode
            ):
                _verify(f"{ckpt_region}->{eval_region}/{kind}", final, row)
                return
    raise VerificationError(
        f"No transfer_matrix_raw.csv row for {kind} {ckpt_region}->{eval_region} ep{episode}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", default="results/wildfire_phase3_multiseed")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument(
        "--skip-unverified",
        action="store_true",
        help="skip (with a loud warning) rollouts whose full row does not match the frozen "
        "record, instead of aborting; skips are documented in index.json",
    )
    args = ap.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    index: list[dict] = []
    skipped: list[dict] = []

    def write(payload: dict, name: str, kind_tag: str) -> None:
        path = out_dir / name
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        m = payload["meta"]
        index.append(
            {
                "file": name,
                "kind": kind_tag,
                **{
                    k: m[k]
                    for k in ("region", "policy", "ckpt_region", "episode_seed", "steps", "final")
                },
            }
        )
        print(
            f"  wrote {name} ({path.stat().st_size / 1024:.0f} KB, {m['steps']} steps, "
            f"WEL {m['final']['WEL']}, ISR {round(m['final']['ISR'], 4)}) [verified]"
        )

    def skip(err: VerificationError, region: str, kind: str, kind_tag: str) -> None:
        if not args.skip_unverified:
            raise SystemExit(
                f"VERIFICATION FAILED {err}\n(re-run with --skip-unverified "
                f"to export the rollouts that do verify)"
            )
        print(f"  SKIPPED (unverified): {err}")
        skipped.append({"region": region, "policy": kind, "kind": kind_tag, "reason": str(err)})

    for region in regions:
        for kind in policies:
            print(f"{region}/{kind} …", flush=True)
            payload = capture_rollout(
                region, kind, region, args.seed, args.episode, args.ckpt_dir, device
            )
            try:
                verify_native(payload["meta"]["final"], region, kind, args.seed, args.episode)
            except VerificationError as err:
                skip(err, region, kind, "native")
                continue
            write(payload, f"{region}_{kind}_s{args.seed}.json", "native")

    if not args.no_transfer:
        for eval_region in regions:
            other = "california" if eval_region == "saudi" else "saudi"
            for kind in TRANSFER_POLICIES:
                if kind not in policies:
                    continue
                print(f"{other}->{eval_region}/{kind} (transfer) …", flush=True)
                payload = capture_rollout(
                    eval_region, kind, other, args.seed, args.episode, args.ckpt_dir, device
                )
                try:
                    verify_transfer(
                        payload["meta"]["final"], eval_region, other, kind, args.seed, args.episode
                    )
                except VerificationError as err:
                    skip(err, eval_region, kind, "transfer")
                    continue
                write(payload, f"{eval_region}_{kind}_from_{other}_s{args.seed}.json", "transfer")

    (out_dir / "index.json").write_text(
        json.dumps({"replays": index, "skipped_unverified": skipped}, indent=1) + "\n"
    )
    print(f"\n{len(index)} replays -> {out_dir} (each verified: full frozen-row match)")
    if skipped:
        print(
            f"{len(skipped)} rollout(s) SKIPPED as unverified — investigate before "
            f"camera-ready: {[s['policy'] + '/' + s['region'] for s in skipped]}"
        )


if __name__ == "__main__":
    main()
