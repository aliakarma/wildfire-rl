import json

s = json.load(open("results/phase3_peer/ablation_summary_saudi.json"))
print("REGION:", s["region"], "| unit:", s["protocol"]["unit"])
print("full-system source:", s["protocol"]["full_system_source"])
for metric in ["WEL", "ISR"]:
    print(f"--- {metric} ---")
    for pol, st in s["metrics"][metric].items():
        print(
            f"  {pol:32s} mean {st['mean']:8.3f}  [{st['ci_lo']:.2f}, {st['ci_hi']:.2f}]"
            f"  per-seed {st['per_seed']}"
        )
print("--- WEL comparisons vs Full System (shipped ckpt), Welch on 5 seed means ---")
for pol, c in s["comparisons_vs_full_shipped"]["WEL"].items():
    if c["p_val"] is None:
        print(f"  {pol:32s} {c.get('note', 'degenerate')}")
    else:
        print(f"  {pol:32s} p={c['p_val']:.4g}  d={c['cohens_d']:.2f}")
