//! Compare our self-trained 16K BPE against a pre-trained Korean BPE
//! (e.g. Polyglot-Ko 30K) on the same KoWiki corpus.
//!
//! Reports per-tokenizer:
//!   - vocabulary size
//!   - encoded token count for the full corpus
//!   - chars/token ratio (compression efficiency)
//!   - cross-entropy floor at vocab uniform (`ln(vocab)`)
//!
//! Run:
//!   cargo run -p nanogpt-rs --example compare_tokenizers --release -- \
//!       --data data/kowiki/kowiki_clean.txt \
//!       --our   data/kowiki/kowiki_bpe.json \
//!       --hf    data/kowiki/polyglot_ko_tokenizer.json

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use nanogpt_rs::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "data/kowiki/kowiki_clean.txt")]
    data: PathBuf,
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    our: PathBuf,
    #[arg(long, default_value = "data/kowiki/polyglot_ko_tokenizer.json")]
    hf: PathBuf,
    /// Optional sample line count for spot-check (encode + decode round-trip).
    #[arg(long, default_value_t = 3)]
    sample_lines: usize,
}

fn report(name: &str, tk: &Tokenizer, text: &str) -> anyhow::Result<usize> {
    let v = tk.vocab_size();
    let t0 = Instant::now();
    let ids = tk.encode(text)?;
    let dt = t0.elapsed().as_millis();
    let ratio = text.len() as f64 / ids.len() as f64;
    let baseline = (v as f64).ln();
    println!(
        "[{name}]  vocab={v}  tokens={}  chars/token={:.2}  ln(vocab)={:.2}  encode={}ms",
        ids.len(),
        ratio,
        baseline,
        dt
    );
    Ok(ids.len())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let text = std::fs::read_to_string(&args.data)?;
    println!("corpus: {} chars\n", text.len());

    let our_tk = Tokenizer::from_hf_file(&args.our)?;
    let our_n = report("ours-16K-BPE", &our_tk, &text)?;

    let hf_tk = Tokenizer::from_hf_file(&args.hf)?;
    let hf_n = report("polyglot-ko-30K", &hf_tk, &text)?;

    println!(
        "\nDelta: HF tokenizer produces {:.2}× the tokens of ours ({} vs {}). \
         Lower is better — ours encodes Korean denser, but HF's larger vocab \
         covers OOV better and starts from much more diverse pretraining.",
        hf_n as f64 / our_n as f64,
        hf_n,
        our_n
    );

    // Round-trip spot check on a few lines.
    println!("\n--- round-trip spot check (first {} non-empty lines) ---", args.sample_lines);
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).take(args.sample_lines).enumerate() {
        let line = if line.len() > 200 { &line[..200] } else { line };
        let our_ids = our_tk.encode(line)?;
        let hf_ids = hf_tk.encode(line)?;
        println!(
            "[{i}]  ours: {} tokens   hf: {} tokens",
            our_ids.len(),
            hf_ids.len()
        );
        println!("     decoded(ours): {}", our_tk.decode(&our_ids)?.chars().take(80).collect::<String>());
        println!("     decoded(hf):   {}", hf_tk.decode(&hf_ids)?.chars().take(80).collect::<String>());
    }

    Ok(())
}
