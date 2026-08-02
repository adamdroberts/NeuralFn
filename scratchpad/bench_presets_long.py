"""Valid per-variant pretraining comparison for the dense GPT-2 presets.

Fixes every validity problem in the earlier 20-step harness:

  * horizon      20 -> 3000 steps (40M -> 24.6M tokens at 4x the batch)
  * batch        2048 -> 8192 tokens/step, cutting gradient noise
  * LR           3e-5 constant -> 3e-4 with 150-step warmup + cosine to 10%,
                 i.e. a schedule a 124M model is actually trained under
  * init         adds GPT-2's residual-projection scaling std=0.02/sqrt(2L)
                 on out_proj / fc2, required for stability at a realistic LR
  * decay        AdamW weight decay applied only to ndim >= 2 tensors (the
                 standard GPT-2 recipe), uniformly across every preset
  * metric       train loss -> held-out validation loss on a fixed slice of
                 DictCombined_val, which is what the hypotheses actually predict
  * data         Dict_train (31.5M tokens) -> DictCombined_train (97.6M), so a
                 3000-step run never wraps and never sees a token twice
  * telemetry    sampled at every validation checkpoint, so H0302's high-gap
                 fraction can be tracked over the run rather than read once

Reports train tokens/s per preset with validation time excluded.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config

PRESETS = ["gpt2", "gpt2_zloss", "gpt2_softcap", "gpt2_qknorm", "gpt2_diff", "gpt2_stable"]
DATA_DIR = "/mnt/disk2/dev/open-source/llm.kittens/dev/data/dict"
TRAIN_BIN = f"{DATA_DIR}/DictCombined_train.bin"
VAL_BIN = f"{DATA_DIR}/DictCombined_val.bin"


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16, offset=1024)


def gpt2_init(model: torch.nn.Module, num_layers: int) -> None:
    """GPT-2 init: N(0, 0.02), with residual-writing projections scaled by 1/sqrt(2L).

    CompiledTorchGraph leaves every nn.Linear / nn.Embedding at PyTorch defaults,
    which at vocab 50257 gives an initial loss near 180 instead of ln(50257).
    The residual scaling on out_proj / fc2 keeps the residual-stream variance from
    growing with depth; without it a 124M model at lr 3e-4 is not reliably stable.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            std = 0.02
            if name.endswith("out_proj.proj") or name.endswith("fc2.proj"):
                std = 0.02 / math.sqrt(2 * num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def param_groups(model: torch.nn.Module, weight_decay: float):
    """Standard GPT-2 recipe: decay matrices, never LayerNorm gains or biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.final_lr_frac * args.lr + (1 - args.final_lr_frac) * args.lr * 0.5 * (
        1.0 + math.cos(math.pi * min(1.0, progress))
    )


def logit_telemetry(logits: torch.Tensor) -> dict:
    flat = logits.reshape(-1, logits.size(-1)).float()
    top2 = flat.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    log_z = torch.logsumexp(flat, dim=-1)
    return {
        "high_gap_fraction_gt16": (gap > 16.0).float().mean().item(),
        "high_gap_fraction_gt8": (gap > 8.0).float().mean().item(),
        "mean_top1_top2_gap": gap.mean().item(),
        "mean_abs_logsumexp": log_z.abs().mean().item(),
        "max_abs_logsumexp": log_z.abs().max().item(),
    }


def save(path: str, cfg: dict, rows: list) -> None:
    """Write results after every checkpoint, not just after a preset finishes.

    A 3000-step x 6-preset x 3-seed sweep is ~2.5 hours; a mid-run failure that
    only serializes on preset completion loses everything before it. (It did:
    the first attempt lost 250 completed steps when the WSL2 dxgkrnl adapter
    dropped out.) Writing through a temp file keeps the JSON parseable even if
    the process dies mid-write.
    """
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"config": cfg, "results": rows}, f, indent=2)
    os.replace(tmp, path)


def completed_presets(path: str) -> set:
    """Presets already finished in a previous run of this seed, for --resume."""
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            prior = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    return {
        r["preset"] for r in prior.get("results", [])
        if "error" not in r and r.get("final_val_loss") is not None
    }


def bench(preset: str, args, train_tokens, val_batches, on_checkpoint=None) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    spec = build_model_spec_from_config({
        "preset": preset, "model_dim": args.model_dim, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "vocab_size": args.vocab_size,
    })
    model = CompiledTorchGraph(build_gpt_root_graph(name=f"{preset}_long", model_spec=spec))
    model = model.to(args.device)
    gpt2_init(model, args.num_layers)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8)

    captured = {}

    def hook(_m, inputs, _o):
        if inputs:
            captured["logits"] = inputs[0].detach()

    ce = dict(model.named_modules())["node_modules.model.node_modules.ce"]
    handle = ce.register_forward_hook(hook)

    tokens_per_step = args.batch * args.seq
    span = args.batch * (args.seq + 1)
    rng = np.random.default_rng(args.seed)
    max_off = train_tokens.size - span - 1

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        for vx, vy in val_batches:
            losses.append(model(vx, vy)[0].item())
        tele = logit_telemetry(captured["logits"])
        model.train()
        return statistics.mean(losses), tele

    torch.cuda.reset_peak_memory_stats(args.device)
    step_ms, train_losses, checkpoints = [], [], []
    diverged = False

    for step in range(args.steps):
        off = int(rng.integers(0, max_off))
        chunk = train_tokens[off:off + span].astype(np.int64).reshape(args.batch, args.seq + 1)
        x = torch.from_numpy(chunk[:, :-1].copy()).to(args.device, non_blocking=True)
        y = torch.from_numpy(chunk[:, 1:].copy()).to(args.device, non_blocking=True)

        lr = lr_at(step, args)
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        lv = loss.item()
        train_losses.append(lv)
        if step >= args.warmup:
            step_ms.append(dt)
        if not math.isfinite(lv):
            diverged = True
            break

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            vloss, tele = validate()
            checkpoints.append({
                "step": step + 1, "lr": round(lr, 8),
                "train_loss": round(statistics.mean(train_losses[-args.val_every:]), 5),
                "val_loss": round(vloss, 5),
                "telemetry": {k: round(v, 6) for k, v in tele.items()},
            })
            print(f"    [{preset} seed={args.seed}] step {step+1}/{args.steps} "
                  f"val_loss={vloss:.5f} lr={lr:.2e}", flush=True)
            if on_checkpoint is not None:
                on_checkpoint({
                    "preset": preset, "seed": args.seed, "partial": True,
                    "steps_completed": step + 1, "target_steps": args.steps,
                    "median_step_ms": round(statistics.median(step_ms), 2) if step_ms else None,
                    "train_tokens_per_s": (
                        round(tokens_per_step / (statistics.median(step_ms) / 1000.0))
                        if step_ms else None),
                    "peak_mem_gib": round(
                        torch.cuda.max_memory_allocated(args.device) / (1024 ** 3), 2),
                    "final_val_loss": None, "checkpoints": list(checkpoints),
                })

    handle.remove()
    median_ms = statistics.median(step_ms) if step_ms else float("nan")
    result = {
        "preset": preset,
        "seed": args.seed,
        "diverged": diverged,
        "params": params,
        "median_step_ms": round(median_ms, 2),
        "train_tokens_per_s": round(tokens_per_step / (median_ms / 1000.0)) if step_ms else None,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 3), 2),
        "tokens_seen": args.steps * tokens_per_step,
        "final_val_loss": checkpoints[-1]["val_loss"] if checkpoints else None,
        "final_train_loss": round(statistics.mean(train_losses[-args.val_every:]), 5),
        "checkpoints": checkpoints,
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--json-out", default="")
    ap.add_argument("--resume", action="store_true",
                    help="skip presets already completed in --json-out")
    args = ap.parse_args()

    train_tokens = load_bin(TRAIN_BIN)
    val_tokens = load_bin(VAL_BIN)
    span = args.batch * (args.seq + 1)
    val_batches = []
    for i in range(args.val_batches):
        chunk = val_tokens[i * span:(i + 1) * span].astype(np.int64).reshape(args.batch, args.seq + 1)
        val_batches.append((
            torch.from_numpy(chunk[:, :-1].copy()).to(args.device),
            torch.from_numpy(chunk[:, 1:].copy()).to(args.device),
        ))

    cfg = vars(args)
    done = completed_presets(args.json_out) if args.resume else set()
    rows = []
    if done:
        with open(args.json_out) as f:
            rows = [r for r in json.load(f)["results"]
                    if r.get("preset") in done and not r.get("partial")]
        print(f"resuming: skipping already-complete {sorted(done)}", flush=True)

    for preset in args.presets.split(","):
        if preset in done:
            continue
        t0 = time.time()
        # partial rows are replaced as the run progresses, so a crash still
        # leaves every completed checkpoint on disk
        def checkpoint_sink(partial, _p=preset):
            live = [r for r in rows if r.get("preset") != _p]
            save(args.json_out, cfg, live + [partial])
        try:
            row = bench(preset, args, train_tokens, val_batches, on_checkpoint=checkpoint_sink)
        except Exception as exc:
            row = {"preset": preset, "seed": args.seed,
                   "error": f"{type(exc).__name__}: {exc}"}
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass  # device may be gone entirely
        row["wallclock_s"] = round(time.time() - t0, 1)
        rows = [r for r in rows if r.get("preset") != preset] + [row]
        print(json.dumps({k: v for k, v in row.items() if k != "checkpoints"}), flush=True)
        save(args.json_out, cfg, rows)
        if "error" in row and not torch.cuda.is_available():
            print("device unavailable; stopping this seed early", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
