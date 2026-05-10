"""Phase 16 S4 — Hybrid OPD+SFT student trainer.

Phase 15 S2 (forward-KL) and Phase 16 S2 (reverse-KL) both retracted
multi-teacher OPD as destructive at LoRA scale. Most plausible
remaining mechanism: KL alone has no anchor to verifier-passed
completions; ANY KL update can drift student to high-entropy or
mode-collapsed regions depending on direction.

Hybrid OPD+SFT mixes:
  loss = (1 − α) · OPD_KL + α · SFT_NLL_chosen

α=0.3 is the Phase 11 S5 hybrid-DPO winner; same value used here as
a default. SFT term anchors the student to verifier-passed
completions while OPD KL provides specialist supervision.

If hybrid lifts mean OR tightens σ vs Phase 16 S2 reverse-KL, OPD-as-
regularizer is salvageable. If hybrid still LOSES vs SFT-only (Phase
15 S1 baseline), OPD is fundamentally unhelpful at this scale.

Reuses Phase 15 S2 specialists in checkpoints/phase15_s2/.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s1"))
from self_improve import harvest_round  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "phase14_c4"))
from opd import opd_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "phase15_s2"))
from routing import SUBSETS, classify  # noqa: E402
from self_improve_opd import OpdPairDataset, opd_collate  # noqa: E402


def hybrid_opd_finetune(student_model, teachers, tokenizer, triples,
                        steps, batch_size, lr, device, temperature,
                        kl_direction, sft_alpha):
    """Hybrid OPD+SFT: loss = (1-α)·OPD_KL + α·SFT_NLL_chosen.

    α=0 → pure OPD (= Phase 15 S2)
    α=1 → pure SFT (= Phase 15 S1)
    α∈(0,1) → mix
    """
    if not triples:
        return 0.0

    pad_id = tokenizer.eos_token_id
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
        subset = subset_keys[step % len(subset_keys)]
        try:
            batch = next(iters[subset])
        except StopIteration:
            iters[subset] = iter(loaders[subset])
            batch = next(iters[subset])
        batch.pop("subset")
        batch = {k: v.to(device) for k, v in batch.items()}

        # Teacher logits (frozen)
        teacher = teachers[subset]
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            teacher_logits = t_out.logits.detach()
        # Student logits
        s_out = student_model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"])
        student_logits = s_out.logits

        # OPD KL component
        opd_kl = opd_loss(
            student_logits, [(1.0, teacher_logits)], batch["labels"],
            temperature=temperature, direction=kl_direction,
        )
        # SFT NLL component (standard causal-LM cross-entropy at chosen positions)
        shift_logits = student_logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        sft_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        loss = (1.0 - sft_alpha) * opd_kl + sft_alpha * sft_nll
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
    ap.add_argument("--specialists-dir", default="checkpoints/phase15_s2")
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
    ap.add_argument("--opd-temperature", type=float, default=2.0)
    ap.add_argument("--kl-direction", choices=["forward", "reverse"], default="reverse",
                    help="reverse default (DeepSeek V4 choice; forward also tested)")
    ap.add_argument("--sft-alpha", type=float, default=0.3,
                    help="SFT mixing weight (Phase 11 S5 hybrid-DPO winner; 0=pure OPD, 1=pure SFT)")
    ap.add_argument("--verify-timeout", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = args.out or (
        f"/raid/users/paul/workLLM/scripts/phase16_s4/"
        f"run_hybrid_a{args.sft_alpha}_kl{args.kl_direction}_seed{args.seed}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P16S4-Hybrid] seed={args.seed} α={args.sft_alpha} kl={args.kl_direction} T={args.opd_temperature}")

    all_challenges = []
    for items in SUBSETS.values():
        all_challenges.extend(items)
    all_challenges.sort(key=lambda c: c["name"])

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    print(f"[P16S4-Hybrid] student loaded in {time.time() - t0:.1f}s")

    teachers = {}
    for subset in SUBSETS.keys():
        adapter_dir = Path(args.specialists_dir) / f"specialist_{subset}"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"specialist adapter missing: {adapter_dir}")
        t1 = time.time()
        base_t = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(device)
        t_model = PeftModel.from_pretrained(base_t, str(adapter_dir))
        t_model.eval()
        for p in t_model.parameters():
            p.requires_grad = False
        teachers[subset] = t_model
        print(f"[P16S4-Hybrid] loaded teacher '{subset}' in {time.time() - t1:.1f}s")

    history = []
    seed_base = args.seed * 1_000_000
    for r in range(args.rounds + 1):
        is_post = r == args.rounds
        label = f"round-{r}" if not is_post else f"final-{r}"
        t_round = time.time()
        print(f"\n========== seed={args.seed} hybrid α={args.sft_alpha} {label} ==========")
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
        ch_prompt = {ch["name"]: ch["prompt"] for ch in all_challenges}
        triples = []
        for rec in records:
            if not rec["verdict"]:
                continue
            prompt = ch_prompt[rec["challenge"]]
            subset = classify(prompt)
            triples.append((prompt, rec["completion"], subset))
        if len(triples) < 2:
            print("  [WARN] too few verifier-passed; skipping hybrid step")
            continue
        t = time.time()
        loss = hybrid_opd_finetune(
            student, teachers, tokenizer, triples,
            steps=args.train_steps, batch_size=args.batch_size,
            lr=args.lr, device=device,
            temperature=args.opd_temperature,
            kl_direction=args.kl_direction,
            sft_alpha=args.sft_alpha,
        )
        print(f"  Hybrid[α={args.sft_alpha} kl={args.kl_direction}]: "
              f"{len(triples)} pairs × {args.train_steps} steps, "
              f"last_loss={loss:.3f}, {time.time() - t:.1f}s")

    out = {
        "model": args.model, "seed": args.seed,
        "kl_direction": args.kl_direction,
        "opd_temperature": args.opd_temperature,
        "sft_alpha": args.sft_alpha,
        "rounds": args.rounds, "samples": args.samples,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_steps": args.train_steps, "lr": args.lr,
        "history": history,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n[P16S4 seed={args.seed} α={args.sft_alpha}] wrote {out_path}")
    for h in history:
        print(f"  {h['label']:12s}  pass={h['pass_rate']:.3f} ({h['n_pass']}/{h['n']})")


if __name__ == "__main__":
    main()
