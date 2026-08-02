"""Per-variant benchmark for the dense GPT-2 pretraining presets.

Trains each preset for N steps on real gpt2-BPE tokens through the Torch path
(the native C++ loop does not implement these variants yet) and reports:

  * median optimizer-step time, tokens/s, peak device memory, parameter count
  * loss at step 1 and step N
  * the H0302 telemetry the z-loss hypothesis actually predicts:
      - fraction of rows whose top-1 minus top-2 logit gap exceeds 16
      - mean and max |logsumexp(logits)|  (the quantity z-loss anchors)

Every preset sees the identical token stream and identical init seed, so the
comparison is paired.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config

PRESETS = ["gpt2", "gpt2_zloss", "gpt2_softcap", "gpt2_qknorm", "gpt2_diff", "gpt2_stable"]
SHARD = "/mnt/disk2/dev/open-source/llm.kittens/dev/data/dict/Dict_train.bin"


def load_tokens(count: int) -> np.ndarray:
    raw = np.fromfile(SHARD, dtype=np.uint16, count=count + 1024, offset=1024)
    return raw.astype(np.int64)


def make_batches(tokens: np.ndarray, steps: int, batch: int, seq: int, device: str):
    need = steps * batch * (seq + 1)
    assert tokens.size >= need, f"need {need} tokens, have {tokens.size}"
    out = []
    off = 0
    for _ in range(steps):
        chunk = tokens[off: off + batch * (seq + 1)].reshape(batch, seq + 1)
        off += batch * (seq + 1)
        x = torch.from_numpy(chunk[:, :-1].copy()).to(device)
        y = torch.from_numpy(chunk[:, 1:].copy()).to(device)
        out.append((x, y))
    return out


def logit_telemetry(logits: torch.Tensor) -> dict:
    flat = logits.reshape(-1, logits.size(-1)).float()
    top2 = flat.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1])
    log_z = torch.logsumexp(flat, dim=-1)
    return {
        "high_gap_fraction_gt16": (gap > 16.0).float().mean().item(),
        "mean_top1_top2_gap": gap.mean().item(),
        "mean_abs_logsumexp": log_z.abs().mean().item(),
        "max_abs_logsumexp": log_z.abs().max().item(),
    }


def gpt2_init(model: torch.nn.Module) -> None:
    """Apply GPT-2's initialization to the compiled graph.

    CompiledTorchGraph leaves every nn.Linear / nn.Embedding at PyTorch's default
    init, which at vocab 50257 gives logits with std ~sqrt(model_dim) and an
    initial loss around 180 instead of ln(50257) = 10.82. That regime saturates
    every logit-scale metric (high-gap fraction pins at 1.0), so the presets
    cannot be told apart. Re-init to N(0, 0.02) as in the GPT-2 paper.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def bench(preset: str, args, batches) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    spec = build_model_spec_from_config({
        "preset": preset,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "vocab_size": args.vocab_size,
    })
    graph = build_gpt_root_graph(name=f"{preset}_bench", model_spec=spec)
    model = CompiledTorchGraph(graph).to(args.device)
    gpt2_init(model)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)

    captured = {}

    def hook(_mod, inputs, _out):
        if inputs:
            captured["logits"] = inputs[0].detach()

    ce = dict(model.named_modules())["node_modules.model.node_modules.ce"]
    handle = ce.register_forward_hook(hook)

    torch.cuda.reset_peak_memory_stats(args.device)
    step_ms, losses = [], []
    tele_first = tele_last = None

    for i, (x, y) in enumerate(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        losses.append(loss.item())
        if i >= args.warmup:
            step_ms.append(dt)
        if i == 0:
            tele_first = logit_telemetry(captured["logits"])
        if i == len(batches) - 1:
            tele_last = logit_telemetry(captured["logits"])

    handle.remove()
    median_ms = statistics.median(step_ms)
    tokens_per_step = args.batch * args.seq
    peak_gib = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
    result = {
        "preset": preset,
        "params_m": round(params / 1e6, 2),
        "median_step_ms": round(median_ms, 2),
        "tokens_per_s": round(tokens_per_step / (median_ms / 1000.0)),
        "peak_mem_gib": round(peak_gib, 2),
        "loss_step1": round(losses[0], 5),
        "loss_final": round(losses[-1], 5),
        "telemetry_step1": {k: round(v, 6) for k, v in tele_first.items()},
        "telemetry_final": {k: round(v, 6) for k, v in tele_last.items()},
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
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    tokens = load_tokens(args.steps * args.batch * (args.seq + 1) + 16)
    batches = make_batches(tokens, args.steps, args.batch, args.seq, args.device)

    rows = []
    for preset in args.presets.split(","):
        try:
            rows.append(bench(preset, args, batches))
            print(json.dumps(rows[-1]), flush=True)
        except Exception as exc:  # keep going; report the failure explicitly
            rows.append({"preset": preset, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.empty_cache()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"config": vars(args), "results": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
