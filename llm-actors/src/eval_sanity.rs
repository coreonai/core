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

/// One published reference point: greedy full-set base-model pass@1.
#[derive(Debug, Clone, Copy)]
pub struct PublicBaseline {
    /// The `--model-id` suffix this applies to (matched as a substring of the
    /// caller's model id, so a snapshot path or an `-Instruct` variant still
    /// resolves to the right base row when it contains this token).
    pub model_id: &'static str,
    pub benchmark: &'static str,
    pub published_pass1: f64,
    /// Half-width of the accepted band. Generous enough to absorb harness
    /// differences (prompt formatting, stop tokens) yet tight enough to catch
    /// the ~2× mis-measurement class the C5 bug produced.
    pub tol: f64,
    pub source: &'static str,
}

/// Official Qwen2.5-Coder **base**-model greedy full-set pass@1
/// (Qwen2.5-Coder Technical Report, arXiv:2409.12186 v3, Table 5).
pub const PUBLIC_BASELINES: &[PublicBaseline] = &[
    PublicBaseline {
        model_id: "Qwen2.5-Coder-0.5B",
        benchmark: "HumanEval",
        published_pass1: 0.280,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-1.5B",
        benchmark: "HumanEval",
        published_pass1: 0.439,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "HumanEval",
        published_pass1: 0.616,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-0.5B",
        benchmark: "MBPP",
        published_pass1: 0.529,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-1.5B",
        benchmark: "MBPP",
        published_pass1: 0.692,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
    PublicBaseline {
        model_id: "Qwen2.5-Coder-7B",
        benchmark: "MBPP",
        published_pass1: 0.769,
        tol: 0.10,
        source: "arXiv:2409.12186 Table 5",
    },
];

/// Outcome of comparing a measured base pass@1 to the published number.
#[derive(Debug, Clone, PartialEq)]
pub enum SanityVerdict {
    /// Within `tol` of the published value.
    Ok {
        expected: f64,
        tol: f64,
        source: &'static str,
    },
    /// Off by more than `tol` — a likely measurement bug.
    Drift {
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
                expected,
                tol,
                source,
            } => format!(
                "[SANITY] OK — {model_id} {benchmark} base pass@1 = {measured:.4} \
                 vs published {expected:.3}±{tol:.2} ({source})"
            ),
            SanityVerdict::Drift {
                measured,
                expected,
                delta,
                tol,
                source,
            } => format!(
                "[SANITY] DRIFT — {model_id} {benchmark} base pass@1 = {measured:.4} \
                 vs published {expected:.3}±{tol:.2} (off by {delta:+.4}; {source}). \
                 A ~2× miss is the FilteredDomain/truncation bug class — check the \
                 eval path before trusting this number."
            ),
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
            expected: b.published_pass1,
            tol: b.tol,
            source: b.source,
        }
    } else {
        SanityVerdict::Drift {
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
}
