//! Public-baseline sanity check for the eval pipeline.
//!
//! Phase 22 C4/C5 lost several batches and inverted a conclusion because a
//! silent measurement bug (`FilteredDomain` disabled completion truncation)
//! made the 7B base measure **0.246** where the true value is **0.422**, and
//! nothing objected — the number just became a headline. The remedy is to
//! compare an internal *base* measurement against the model's **published**
//! score whenever the two are measured in a comparable way, and flag a drift
//! loudly (see `docs/phase22-c4-c5-rl-vs-sft.md`, Lesson #1/#6).
//!
//! Comparability is the whole game. The published numbers below are
//! **greedy, full-set, completion-style base-model pass@1**. A measurement is
//! only comparable to them in the canonical config: the full benchmark
//! (`n_problems == full`, `offset == 0`), greedy (`passk == 1`), the
//! unfiltered domain, and no trained checkpoint. A filtered/subset/sampled
//! run is *not* comparable and must not be checked against these — that
//! non-comparability is itself a lesson the caller should surface.
//!
//! Two anchor kinds ([`AnchorKind`]): a **cited Point** (HumanEval/MBPP, from
//! the tech report) whose drift is a real red flag, and a **PlausibilityBand**
//! for benchmarks with no comparable public base number (LiveCodeBench is
//! instruct/chat-format-centric, so base+completion isn't on the board). A band
//! only catches a broken harness (≈0) or an implausible score; for those
//! benchmarks the actual contamination signal is the pre-vs-post-cutoff delta
//! (harness-side `--start_date`/`--end_date`), not the band. The two are worded
//! differently in the log so they are never conflated.

/// What kind of anchor a row is — this changes the meaning of the reported
/// numbers, so the eval log must not conflate them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorKind {
    /// A **cited public** leaderboard/report number: `center ± tol` where
    /// `center` is the published value. Drift here is a real red flag.
    Point,
    /// **No comparable public base number exists** (e.g. LiveCodeBench is
    /// instruct/chat-format-centric; base+completion isn't on the board). The
    /// `center ± tol` is a wide **plausibility band** that only catches gross
    /// harness failure (≈0 → broken; implausibly high → wrong problem set), NOT
    /// a leaderboard match. Tighten after the first calibrated run. For such
    /// benchmarks the real contamination signal is the pre-vs-post-cutoff
    /// delta, not this band.
    PlausibilityBand,
}

/// One reference row: a model × benchmark anchor for the base-model pass@1.
#[derive(Debug, Clone, Copy)]
pub struct PublicBaseline {
    /// The `--model-id` suffix this applies to (matched as a substring of the
    /// caller's model id, so a snapshot path or an `-Instruct` variant still
    /// resolves to the right base row when it contains this token).
    pub model_id: &'static str,
    pub benchmark: &'static str,
    pub kind: AnchorKind,
    /// Anchor centre: the published value for a `Point`, or the band midpoint
    /// for a `PlausibilityBand`.
    pub published_pass1: f64,
    /// Half-width of the accepted band. For a `Point`, generous enough to
    /// absorb harness differences yet tight enough to catch the ~2×
    /// mis-measurement class the C5 bug produced; for a `PlausibilityBand`,
    /// deliberately wide.
    pub tol: f64,
    pub source: &'static str,
}

/// Official Qwen2.5-Coder **base**-model greedy full-set pass@1
/// (Qwen2.5-Coder Technical Report, arXiv:2409.12186 v3, Table 5).
pub const PUBLIC_BASELINES: &[PublicBaseline] = &[
    PublicBaseline {
        model_id: "Qwen2.5-Coder-0.5B",
        benchmark: "HumanEval",
        kind: AnchorKind::Point,
        published_pass1: 0.280,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-1.5B",
        benchmark: "HumanEval",
        kind: AnchorKind::Point,
        published_pass1: 0.439,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "HumanEval",
        kind: AnchorKind::Point,
        published_pass1: 0.616,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-0.5B",
        benchmark: "MBPP",
        kind: AnchorKind::Point,
        published_pass1: 0.529,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-1.5B",
        benchmark: "MBPP",
        kind: AnchorKind::Point,
        published_pass1: 0.692,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "MBPP",
        kind: AnchorKind::Point,
        published_pass1: 0.769,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    // LiveCodeBench — no clean public BASE-model point exists (the board is
    // instruct/chat-format-centric; Seed-Coder reports Seed-8B-Instruct 24.7%
    // *surpassing* Qwen2.5-Coder-14B-Instruct, so the 7B base under
    // completion-style prompting sits well below that). These are wide
    // plausibility bands (≈[0.02, 0.45]) that only catch a broken harness
    // (≈0) or an implausible score, NOT a leaderboard match. The real
    // contamination signal is the pre-vs-post-cutoff delta (harness-side
    // --start_date/--end_date), not this band. Tighten `tol` after the first
    // calibrated pre-cutoff run and pin the `--release_version` used.
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "LiveCodeBench",
        kind: AnchorKind::PlausibilityBand,
        published_pass1: 0.13, // band [0.03, 0.23] around the measured base 0.125
        tol: 0.10,
        source: "measured base 0.125 (release_v5 idx 640-760, F32, greedy); guard, not a leaderboard point",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-0.5B",
        benchmark: "LiveCodeBench",
        kind: AnchorKind::PlausibilityBand,
        published_pass1: 0.11, // band midpoint of [0.0, 0.22]
        tol: 0.11,
        source: "plausibility guard — no public base LCB point; calibrate on first run",
    },
    // BigCodeBench (calibrated pass@1) — like LCB, the board is
    // instruct-centric, no clean base+completion point. Reference neighbourhood
    // (Qwen2.5-Coder-7B-*Instruct*, Complete split): Full 41.0%, Hard 18.2%.
    // `Complete` is docstring/completion-style so a base model is *less*
    // depressed than on LCB's chat format, but still below the instruct
    // neighbourhood. Bands guard against a broken Docker harness (≈0 on Full,
    // where a 7B base should score meaningfully) and an implausibly-high score
    // (above the instruct neighbourhood → wrong subset/split). Tighten after
    // the first calibrated run; the benchmark key encodes split+subset because
    // the score depends on both.
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "BigCodeBench-Complete-Full",
        kind: AnchorKind::PlausibilityBand,
        published_pass1: 0.225, // band midpoint of [0.03, 0.42]
        tol: 0.195,
        source: "plausibility guard — instruct neighbourhood 41.0% (bigcode-bench.github.io)",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "BigCodeBench-Complete-Hard",
        kind: AnchorKind::PlausibilityBand,
        published_pass1: 0.17, // band [0.09, 0.25]; recalibrated to measured base 0.169
        tol: 0.08,
        source: "measured base 0.169 (calibrated pass@1, gt 0.973); instruct neighbourhood 18.2% (bigcode-bench.github.io)",
    },
];

/// Outcome of comparing a measured base pass@1 to the published number.
#[derive(Debug, Clone, PartialEq)]
pub enum SanityVerdict {
    /// Within `tol` of the anchor.
    Ok {
        kind: AnchorKind,
        expected: f64,
        tol: f64,
        source: &'static str,
    },
    /// Off by more than `tol` — a likely measurement bug (Point) or a broken
    /// harness / implausible score (PlausibilityBand).
    Drift {
        kind: AnchorKind,
        measured: f64,
        expected: f64,
        delta: f64,
        tol: f64,
        source: &'static str,
    },
    /// No published baseline for this (model, benchmark) — nothing to check.
    NoBaseline,
}

impl SanityVerdict {
    /// True only for [`SanityVerdict::Drift`]. Callers use this to set a
    /// non-zero exit under `--sanity-strict`.
    pub fn is_drift(&self) -> bool {
        matches!(self, SanityVerdict::Drift { .. })
    }

    /// A one-line `[SANITY]`-prefixed description for the eval log.
    pub fn describe(&self, model_id: &str, benchmark: &str, measured: f64) -> String {
        match self {
            SanityVerdict::Ok {
                kind,
                expected,
                tol,
                source,
            } => match kind {
                AnchorKind::Point => format!(
                    "[SANITY] OK — {model_id} {benchmark} base pass@1 = {measured:.4} \
                     vs published {expected:.3}±{tol:.2} ({source})"
                ),
                AnchorKind::PlausibilityBand => format!(
                    "[SANITY] OK (plausible) — {model_id} {benchmark} base pass@1 = {measured:.4} \
                     inside guard band [{:.2}, {:.2}] ({source}); the real signal is the \
                     pre/post-cutoff delta, not this band",
                    expected - tol,
                    expected + tol,
                ),
            },
            SanityVerdict::Drift {
                kind,
                measured,
                expected,
                delta,
                tol,
                source,
            } => match kind {
                AnchorKind::Point => format!(
                    "[SANITY] DRIFT — {model_id} {benchmark} base pass@1 = {measured:.4} \
                     vs published {expected:.3}±{tol:.2} (off by {delta:+.4}; {source}). \
                     A ~2× miss is the FilteredDomain/truncation bug class — check the \
                     eval path before trusting this number."
                ),
                AnchorKind::PlausibilityBand => format!(
                    "[SANITY] IMPLAUSIBLE — {model_id} {benchmark} base pass@1 = {measured:.4} \
                     outside guard band [{:.2}, {:.2}] ({source}). ≈0 means a broken harness \
                     (format/extraction); a high value means the wrong problem set or window — \
                     fix before trusting any delta.",
                    expected - tol,
                    expected + tol,
                ),
            },
            SanityVerdict::NoBaseline => format!(
                "[SANITY] no published baseline for ({model_id}, {benchmark}) — \
                 measured pass@1 = {measured:.4}, not checked"
            ),
        }
    }
}

/// Compare a measured **greedy full-set base** pass@1 against the published
/// number. `model_id` is matched as a substring so a snapshot path or a
/// variant name still resolves to the right base row.
pub fn check_public_baseline(model_id: &str, benchmark: &str, measured: f64) -> SanityVerdict {
    let Some(b) = PUBLIC_BASELINES
        .iter()
        .find(|b| b.benchmark == benchmark && model_id.contains(b.model_id))
    else {
        return SanityVerdict::NoBaseline;
    };
    let delta = measured - b.published_pass1;
    if delta.abs() <= b.tol {
        SanityVerdict::Ok {
            kind: b.kind,
            expected: b.published_pass1,
            tol: b.tol,
            source: b.source,
        }
    } else {
        SanityVerdict::Drift {
            kind: b.kind,
            measured,
            expected: b.published_pass1,
            delta,
            tol: b.tol,
            source: b.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_target_is_ok() {
        // 7B HumanEval published 0.616; a correct pipeline lands within tol.
        let v = check_public_baseline("Qwen2.5-Coder-7B", "HumanEval", 0.60);
        assert!(matches!(v, SanityVerdict::Ok { .. }), "{v:?}");
    }

    #[test]
    fn c5_bug_number_is_drift() {
        // The exact mis-measured base (0.246) must flag against 0.616.
        let v = check_public_baseline("Qwen2.5-Coder-7B", "HumanEval", 0.246);
        assert!(v.is_drift(), "{v:?}");
        if let SanityVerdict::Drift { delta, .. } = v {
            assert!(delta < -0.3, "delta {delta}");
        }
    }

    #[test]
    fn model_id_matched_as_substring() {
        // A snapshot path or variant name still resolves to the base row.
        let v = check_public_baseline(
            "/cache/models--Qwen--Qwen2.5-Coder-0.5B/snapshots/abc",
            "HumanEval",
            0.28,
        );
        assert!(matches!(v, SanityVerdict::Ok { .. }), "{v:?}");
    }

    #[test]
    fn unknown_model_has_no_baseline() {
        let v = check_public_baseline("some-other-model", "HumanEval", 0.5);
        assert_eq!(v, SanityVerdict::NoBaseline);
    }

    #[test]
    fn distinct_sizes_do_not_cross_match() {
        // 0.5B measured at its own 0.28 is OK; checked against 7B's 0.616 it
        // would be a drift — verify the row selection is size-correct.
        let v05 = check_public_baseline("Qwen2.5-Coder-0.5B", "HumanEval", 0.28);
        assert!(matches!(v05, SanityVerdict::Ok { expected, .. } if (expected-0.280).abs()<1e-9));
    }

    #[test]
    fn mbpp_baseline_resolves() {
        let v = check_public_baseline("Qwen2.5-Coder-7B", "MBPP", 0.77);
        assert!(matches!(v, SanityVerdict::Ok { .. }), "{v:?}");
    }

    #[test]
    fn lcb_band_flags_broken_harness_and_accepts_plausible() {
        // A broken LCB harness (extraction/format bug) returns ≈0 → flagged.
        let zero = check_public_baseline("Qwen2.5-Coder-7B", "LiveCodeBench", 0.0);
        assert!(zero.is_drift(), "{zero:?}");
        // A plausible base score sits inside the guard band.
        let ok = check_public_baseline("Qwen2.5-Coder-7B", "LiveCodeBench", 0.15);
        assert!(
            matches!(
                ok,
                SanityVerdict::Ok {
                    kind: AnchorKind::PlausibilityBand,
                    ..
                }
            ),
            "{ok:?}"
        );
        // An implausibly high value (wrong problem set / window) also flags.
        let high = check_public_baseline("Qwen2.5-Coder-7B", "LiveCodeBench", 0.9);
        assert!(high.is_drift(), "{high:?}");
    }

    #[test]
    fn point_and_band_are_worded_differently() {
        let p = check_public_baseline("Qwen2.5-Coder-7B", "HumanEval", 0.60);
        assert!(p
            .describe("Qwen2.5-Coder-7B", "HumanEval", 0.60)
            .contains("vs published"));
        let b = check_public_baseline("Qwen2.5-Coder-7B", "LiveCodeBench", 0.15);
        assert!(b
            .describe("Qwen2.5-Coder-7B", "LiveCodeBench", 0.15)
            .contains("guard band"));
    }

    #[test]
    fn bigcodebench_bands_are_split_subset_keyed_and_guard_extremes() {
        // Broken Docker harness ≈0 on Full flags (a 7B base should score > 0.03).
        let broken = check_public_baseline("Qwen2.5-Coder-7B", "BigCodeBench-Complete-Full", 0.0);
        assert!(broken.is_drift(), "{broken:?}");
        // Plausible base score inside the Full band.
        let ok = check_public_baseline("Qwen2.5-Coder-7B", "BigCodeBench-Complete-Full", 0.20);
        assert!(
            matches!(
                ok,
                SanityVerdict::Ok {
                    kind: AnchorKind::PlausibilityBand,
                    ..
                }
            ),
            "{ok:?}"
        );
        // Above the instruct neighbourhood (>0.42) → implausible (wrong subset?).
        let high = check_public_baseline("Qwen2.5-Coder-7B", "BigCodeBench-Complete-Full", 0.55);
        assert!(high.is_drift(), "{high:?}");
        // Hard has its OWN key/band; 0.35 is fine on Full but above Hard's 0.25.
        let hard_ok = check_public_baseline("Qwen2.5-Coder-7B", "BigCodeBench-Complete-Hard", 0.10);
        assert!(matches!(hard_ok, SanityVerdict::Ok { .. }), "{hard_ok:?}");
        let hard_high =
            check_public_baseline("Qwen2.5-Coder-7B", "BigCodeBench-Complete-Hard", 0.35);
        assert!(
            hard_high.is_drift(),
            "Hard key must use Hard band: {hard_high:?}"
        );
    }
}
