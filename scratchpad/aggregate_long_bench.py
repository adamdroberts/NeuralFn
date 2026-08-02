"""Aggregate the 3000-step preset benchmark across seeds into a report."""
import json
import statistics
import sys

SEEDS = [1337, 2024, 90210]
ORDER = ["gpt2", "gpt2_zloss", "gpt2_softcap", "gpt2_qknorm", "gpt2_diff", "gpt2_stable"]


def main() -> int:
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = []
    for seed in SEEDS:
        try:
            runs.append(json.load(open(f"{base_dir}/long_bench_{seed}.json")))
        except FileNotFoundError:
            print(f"(seed {seed} not present yet)")
    if not runs:
        return 1
    cfg = runs[0]["config"]
    agg = {}
    for run in runs:
        for row in run["results"]:
            if "error" in row:
                print(f"!! {row['preset']} seed={row['seed']}: {row['error']}")
                continue
            agg.setdefault(row["preset"], []).append(row)

    print(f"\nconfig: {cfg['num_layers']}L d={cfg['model_dim']} heads={cfg['num_heads']} "
          f"vocab={cfg['vocab_size']} | batch {cfg['batch']}x{cfg['seq']} "
          f"= {cfg['batch']*cfg['seq']} tok/step | {cfg['steps']} steps "
          f"= {cfg['steps']*cfg['batch']*cfg['seq']/1e6:.1f}M tokens | "
          f"lr {cfg['lr']} warmup {cfg['warmup']} cosine->{cfg['final_lr_frac']:.0%} | "
          f"wd {cfg['weight_decay']} (ndim>=2 only) | seeds {SEEDS}\n")

    presets = [p for p in ORDER if p in agg]
    b = agg.get("gpt2")
    b_ms = statistics.mean(r["median_step_ms"] for r in b) if b else None
    b_val = [r["final_val_loss"] for r in b] if b else None

    print("## Throughput and cost")
    print(f"{'preset':14s} {'ms/step':>8} {'vs gpt2':>8} {'train tok/s':>12} {'vs gpt2':>8} "
          f"{'peak mem':>9} {'params':>12} {'wallclock':>10}")
    for p in presets:
        rows = agg[p]
        ms = statistics.mean(r["median_step_ms"] for r in rows)
        tps = statistics.mean(r["train_tokens_per_s"] for r in rows)
        b_tps = statistics.mean(r["train_tokens_per_s"] for r in b)
        wc = statistics.mean(r["wallclock_s"] for r in rows)
        print(f"{p:14s} {ms:8.2f} {100*(ms-b_ms)/b_ms:+7.1f}% {tps:12,.0f} "
              f"{100*(tps-b_tps)/b_tps:+7.1f}% {rows[0]['peak_mem_gib']:7.2f} GiB "
              f"{rows[0]['params']:12,} {wc:9.0f}s")

    print("\n## Held-out validation loss at 3000 steps")
    print(f"{'preset':14s} {'per seed':>26} {'mean':>9} {'stdev':>8} {'Δ vs gpt2':>10} {'verdict':>18}")
    b_sd = statistics.stdev(b_val) if b and len(b_val) > 1 else 0.0
    for p in presets:
        vals = [r["final_val_loss"] for r in agg[p]]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        delta = m - statistics.mean(b_val)
        pooled = max(b_sd, sd)
        if p == "gpt2":
            verdict = "baseline"
        elif abs(delta) > 3 * pooled:
            verdict = "better" if delta < 0 else "worse"
        elif abs(delta) > pooled:
            verdict = "suggestive" if delta < 0 else "suggestive(worse)"
        else:
            verdict = "within noise"
        print(f"{p:14s} {'/'.join(f'{v:.4f}' for v in vals):>26} {m:9.4f} {sd:8.4f} "
              f"{delta:+10.4f} {verdict:>18}")
    print(f"\nbaseline seed stdev = {b_sd:.4f}; 'better'/'worse' requires |Δ| > 3x the larger stdev")

    print("\n## H0302 metric: fraction of rows with top1-top2 logit gap > 16")
    steps = [c["step"] for c in agg[presets[0]][0]["checkpoints"]]
    print(f"{'preset':14s} " + " ".join(f"{s:>9}" for s in steps))
    for p in presets:
        vals = []
        for i in range(len(steps)):
            vals.append(statistics.mean(
                r["checkpoints"][i]["telemetry"]["high_gap_fraction_gt16"] for r in agg[p]))
        print(f"{p:14s} " + " ".join(f"{v:9.5f}" for v in vals))

    print("\n## gap > 8 (lower threshold, activates earlier)")
    for p in presets:
        vals = []
        for i in range(len(steps)):
            vals.append(statistics.mean(
                r["checkpoints"][i]["telemetry"]["high_gap_fraction_gt8"] for r in agg[p]))
        print(f"{p:14s} " + " ".join(f"{v:9.5f}" for v in vals))

    print("\n## mean |logsumexp| (the quantity z-loss anchors)")
    for p in presets:
        vals = []
        for i in range(len(steps)):
            vals.append(statistics.mean(
                r["checkpoints"][i]["telemetry"]["mean_abs_logsumexp"] for r in agg[p]))
        print(f"{p:14s} " + " ".join(f"{v:9.4f}" for v in vals))

    print("\n## max |logsumexp|")
    for p in presets:
        vals = []
        for i in range(len(steps)):
            vals.append(statistics.mean(
                r["checkpoints"][i]["telemetry"]["max_abs_logsumexp"] for r in agg[p]))
        print(f"{p:14s} " + " ".join(f"{v:9.4f}" for v in vals))

    print("\n## validation loss curve (mean over seeds)")
    print(f"{'preset':14s} " + " ".join(f"{s:>9}" for s in steps))
    for p in presets:
        vals = []
        for i in range(len(steps)):
            vals.append(statistics.mean(r["checkpoints"][i]["val_loss"] for r in agg[p]))
        print(f"{p:14s} " + " ".join(f"{v:9.4f}" for v in vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
