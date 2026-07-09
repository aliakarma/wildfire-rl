import json

for region in ["saudi", "california"]:
    s = json.load(open(f"results/phase2_peer/eval_summary_{region}.json"))
    print("=" * 78)
    print("REGION:", region.upper())
    for metric in ["WEL", "ISR"]:
        print(f"--- {metric} ---")
        for pol, st in s["metrics"][metric].items():
            print(
                f"  {pol:24s} mean {st['mean']:8.3f}  "
                f"[{st['ci_lo']:.2f}, {st['ci_hi']:.2f}]  per-seed {st['per_seed']}"
            )
    print("--- WEL comparisons (Welch on 5 seed means) ---")
    for pol, c in s["comparisons"]["WEL"].items():
        vn, vh = c["vs_NoOp"], c["vs_Hierarchical"]
        def fmt(x):
            if x["p_val"] is None:
                return "degenerate"
            return f"p={x['p_val']:.4g} d={x['cohens_d']:.2f}" if x["cohens_d"] is not None else f"p={x['p_val']:.4g}"
        print(f"  {pol:24s} vs No-Op: {fmt(vn):26s} vs Hier: {fmt(vh)}")

print("=" * 78)
print("BEHAVIOR (treat fraction | mean Chebyshev distance to nearest asset)")
for region in ["saudi", "california"]:
    b = json.load(open(f"results/phase2_peer/behavior_{region}.json"))
    print(f"--- {region} ---")
    for pol, st in b.items():
        print(
            f"  {pol:24s} treat {st['action_fractions'][5]:.3f} | "
            f"dist {st['mean_chebyshev_dist_to_nearest_asset']}"
        )
