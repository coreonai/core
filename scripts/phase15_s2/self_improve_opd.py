"""Phase 15 S2 — unified student trained via offline OPD distillation
from k=3 frozen specialist teachers (one per HumanEval routing
subset).

For each round:
  1. Student harvests rollouts on all 164 problems (LoRA enabled).
  2. Filter to verifier-passed pairs.
  3. Bucket pairs by routing subset → which specialist owns the
     prompt.
  4. LoRA-FT student: for each pair, KL-distill student logits
     toward the routed specialist's logits at completion-token
     positions (offline OPD; teacher logits computed once, frozen).

Compared to true on-policy OPD, this uses harvested chosen pairs
rather than fresh rollouts at training time. Avoids the cost of
re-rolling per step but gives up the "student's own distribution"
property. First-cut acceptable for substrate test.

Reference: scripts/phase14_c4/opd.py.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "phase14_c4"))
from opd import opd_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from routing import SUBSETS, classify  # noqa: E402


class OpdPairDataset(Dataset):
    """Per-pair dataset: stores (prompt, completion, subset_name)."""

    def __init__(self, triples, tokenizer, max_len=512):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        prompt, completion, subset = self.triples[i]
        full = prompt + completion + "\n"
        enc = self.tok(full, return_tensors="pt", truncation=True, max_length=self.max_len)
        ids = enc.input_ids[0]
        prompt_ids = self.tok(prompt, return_tensors="pt", truncation=True,
                              max_length=self.max_len).input_ids[0]
        labels = ids.clone()
        labels[: prompt_ids.shape[0]] = -100
        return {"input_ids": ids, "labels": labels, "subset": subset}


def opd_collate(batch, pad_id):
    # Group batch by subset for single-teacher-per-batch training.
    # The DataLoader is built with subset-sorted indices so each batch
    # is homogeneous.
    subsets = {b["subset"] for b in batch}
    if len(subsets) != 1:
        raise RuntimeError(f"opd_collate: mixed-subset batch {subsets}; "
                           "use subset-sorted sampler")
    subset = next(iter(subsets))
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn,
            "subset": subset}


def opd_finetune(student_model, teachers, tokenizer, triples,
                 steps, batch_size, lr, device, temperature, kl_direction):
    """OPD LoRA-FT on student using subset-routed teacher logits.

    teachers: dict[subset_name → frozen PeftModel]

    Per-subset DataLoaders avoid mixed-subset batches at subset
    boundaries that a single sorted DataLoader produces. Round-robin
    over subsets per training step weights all subsets equally
    regardless of their pair count (avoids the dominant subset
    monopolizing the gradient).
    """
    if not triples:
        return 0.0

    pad_id = tokenizer.eos_token_id
    # One DataLoader per subset — guarantees homogeneous batches.
    loaders = {}
    for subset in {t[2] for t in triples}:
        sub_triples = [t for t in triples if t[2] == subset]
        if not sub_triples:
            continue
        sub_ds = OpdPairDataset(sub_triples, tokenizer)
        loaders[subset] = DataLoader(
            sub_ds, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b, pid=pad_id: opd_collate(b, pid),
        )

    iters = {s: iter(l) for s, l in loaders.items()}
    subset_keys = list(loaders.keys())

    trainable = [p for p in student_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, steps // 10), num_training_steps=steps,
    )
    student_model.train()
    last = 0.0
    for step in range(steps):
        # Round-robin subset selection — even per-subset weighting
        subset = subset_keys[step % len(subset_keys)]
        try:
            batch = next(iters[subset])
        except StopIteration:
            iters[subset] = iter(loaders[subset])
            batch = next(iters[subset])
        batch.pop("subset")
        batch = {k: v.to(device) for k, v in batch.items()}
        # Teacher logits (frozen, no grad)
        teacher = teachers[subset]
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            teacher_logits = t_out.logits.detach()
        # Student logits (with grad)
        s_out = student_model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"])
        student_logits = s_out.logits
        loss = opd_loss(
            student_logits, [(1.0, teacher_logits)], batch["labels"],
            temperature=temperature, direction=kl_direction,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last = loss.item()
    student_model.eval()
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--specialists-dir", default="checkpoints/phase15_s2",
                    help="Dir containing specialist_<subset>/ subdirs with adapters")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opd-temperature", type=float, default=2.0,
                    help="KL softmax temperature (DeepSeek default 2.0)")
    ap.add_argument("--kl-direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase15_s2/run_opd_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P15S2-OPD] seed={args.seed} kl_dir={args.kl_direction} T={args.opd_temperature}")

    # Build set of all challenges (union of subsets) with their routing
    all_challenges = []
    for items in SUBSETS.values():
        all_challenges.extend(items)
    all_challenges.sort(key=lambda c: c["name"])  # stable order

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Student: base + fresh LoRA
    base_student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(base_student, lora_cfg)
    student.print_trainable_parameters()
    print(f"[P15S2-OPD] student loaded in {time.time() - t0:.1f}s")

    # Teachers: one PeftModel per specialist subset
    teachers = {}
    for subset in SUBSETS.keys():
        adapter_dir = Path(args.specialists_dir) / f"specialist_{subset}"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"specialist adapter missing: {adapter_dir}")
        t1 = time.time()
        # Each teacher gets its own base (PEFT can't easily share base
        # with student-being-trained without adapter swapping). 4× memory
        # but simpler and 4× Qwen 0.5B fits comfortably on A100 40GB.
        base_t = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(device)
        t_model = PeftModel.from_pretrained(base_t, str(adapter_dir))
        t_model.eval()
        for p in t_model.parameters():
            p.requires_grad = False
        teachers[subset] = t_model
        print(f"[P15S2-OPD] loaded teacher '{subset}' in {time.time() - t1:.1f}s")

    history = []
    seed_base = args.seed * 1_000_000
    for r in range(args.rounds + 1):
        is_post = r == args.rounds
        label = f"round-{r}" if not is_post else f"final-{r}"
        t_round = time.time()
        print(f"\n========== seed={args.seed} OPD {label} ==========")
        records = harvest_round(
            student, tokenizer, all_challenges, args.samples, seed_base,
            args.max_new_tokens, args.temperature, args.top_p,
            args.verify_timeout,
        )
        seed_base += 100 * len(all_challenges)
        n = len(records)
        n_pass = sum(1 for r_ in records if r_["verdict"])
        rate = n_pass / n if n else 0.0
        per_ch = defaultdict(lambda: [0, 0])
        for rec in records:
            per_ch[rec["challenge"]][1] += 1
            if rec["verdict"]:
                per_ch[rec["challenge"]][0] += 1
        print(f"  total={n} pass={n_pass} ({rate:.3f})  {time.time() - t_round:.1f}s")
        history.append({
            "label": label, "n": n, "n_pass": n_pass, "pass_rate": rate,
            "per_challenge": {k: {"pass": v[0], "total": v[1]} for k, v in per_ch.items()},
        })
        if is_post:
            break
        # Build subset-routed triples for OPD-FT
        ch_prompt = {ch["name"]: ch["prompt"] for ch in all_challenges}
        triples = []
        for rec in records:
            if not rec["verdict"]:
                continue
            prompt = ch_prompt[rec["challenge"]]
            subset = classify(prompt)
            triples.append((prompt, rec["completion"], subset))
        per_subset_count = {s: sum(1 for t in triples if t[2] == s) for s in SUBSETS}
        print(f"  routed pairs: {per_subset_count}")
        if len(triples) < 2:
            print("  [WARN] too few verifier-passed; skipping OPD step")
            continue
        t = time.time()
        loss = opd_finetune(student, teachers, tokenizer, triples,
                            steps=args.train_steps, batch_size=args.batch_size,
                            lr=args.lr, device=device,
                            temperature=args.opd_temperature,
                            kl_direction=args.kl_direction)
        print(f"  OPD-FT: {len(triples)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed,
        "kl_direction": args.kl_direction, "opd_temperature": args.opd_temperature,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P15S2-OPD seed={args.seed}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
