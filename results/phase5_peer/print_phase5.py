import json

print("=" * 82)
print("GENERALIZATION SUITE — dWEL vs same-condition No-Op (>0 = policy adds value)")
for region in ["saudi", "california"]:
    s = json.load(open(f"results/phase5_peer/generalization_summary_{region}.json"))
    print(f"\n### REGION: {region.upper()}")
    for cond, pols in s["conditions"].items():
        print(f"  -- {cond} --")
        for pol, e in pols.items():
            dwel = e["dWEL_vs_noop"]
            wel = e["WEL_mean"]
            isr = e["ISR_mean"]
            print(f"     {pol:34s} WEL {wel:6.2f}  ISR {isr:.3f}  dWEL {dwel:+6.2f}")

print("\n" + "=" * 82)
print("SCALE STUDY — WEL by agent count")
for region in ["saudi", "california"]:
    s = json.load(open(f"results/phase5_peer/scale_summary_{region}.json"))
    print(f"\n### REGION: {region.upper()}")
    for n, pols in s["by_num_agents"].items():
        print(f"  -- N={n} --")
        for pol, e in pols.items():
            print(f"     {pol:24s} WEL {e['WEL_mean']:6.2f}  ISR {e['ISR_mean']:.3f}")
