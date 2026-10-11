import json

with open(r"C:\Users\TANISH PAREKH\Downloads\exp4_cbm_full_benchmark.json") as f:
    d = json.load(f)

conds = d["sweep_75"]["conditions"]
harm_cases = [c for c in conds if c["gain"] < 0]
print(f"Total conditions: {len(conds)}")
print(f"Harmed conditions: {len(harm_cases)}")
for h in harm_cases:
    diff = h["gain"] * 100
    print(f"{h['corruption']} Sev {h['severity']}: Static {h['static_mean']*100:.2f}% -> Adapt {h['adapt_mean']*100:.2f}% ({diff:+.2f}%)")

print("\n--- TOP 10 LARGEST GAINS ---")
sorted_gains = sorted(conds, key=lambda x: x["gain"], reverse=True)[:10]
for g in sorted_gains:
    diff = g["gain"] * 100
    print(f"{g['corruption']:20s} Sev {g['severity']}: Static {g['static_mean']*100:.2f}% -> Adapt {g['adapt_mean']*100:.2f}% ({diff:+.2f}%)")

