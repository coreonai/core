//! Streaming extractor for a Wikipedia XML dump.
//!
//! Reads the dump from stdin (decompressed XML — typically piped from
//! `bzcat kowiki-latest-pages-articles.xml.bz2`) and writes one cleaned
//! plain-text article per line to stdout. Wiki markup is reduced to plain
//! text via a small set of regex-based rules; the goal is "good enough
//! corpus for BPE + LM training", not perfect rendering.
//!
//! Run:
//!   bzcat data/kowiki/kowiki-latest-pages-articles.xml.bz2 \
//!     | cargo run -p nanogpt-rs --example extract_kowiki --release \
//!     > data/kowiki/kowiki_plain.txt

use std::io::{self, BufWriter, Write};

use quick_xml::events::Event;
use quick_xml::reader::Reader;
use regex::Regex;

struct Cleaner {
    /// `{{...}}` templates (handle nesting iteratively).
    template: Regex,
    /// `[[File:...]]`, `[[Image:...]]`, `[[Category:...]]` (and Korean variants).
    media: Regex,
    /// `[[link|text]]` → `text`.
    link_with_text: Regex,
    /// `[[link]]` → `link`.
    link_plain: Regex,
    /// `<ref ...>...</ref>` and `<ref .../>`.
    ref_tag: Regex,
    /// Generic remaining `<tag>...</tag>` and `<tag/>`.
    html_tag: Regex,
    /// `[https://... text]` and bare `[https://...]` external links.
    external_link: Regex,
    /// Match a section header line so we can decide to keep or drop it.
    header_line: Regex,
    /// Wiki headers `===Header===`, `==Header==` etc — keep text only after
    /// section-dropping has already happened.
    header_strip: Regex,
    /// Wiki tables: `{| ... |}` non-greedy.
    table: Regex,
    /// Stray table syntax that survives partial markup: `|}`, `|-`, `|+`,
    /// lines that consist of a leading `|` and pipe-separated table cells.
    table_residue: Regex,
    /// `'''bold'''` / `''italic''` markers — strip the quotes, keep text.
    bold: Regex,
    italic: Regex,
    /// Bullet/numbered list markers at line start.
    list_marker: Regex,
    /// Multiple blank lines → one.
    blank_lines: Regex,
    /// Lines that look meaningless (just punctuation / very short).
    noise_line: Regex,
}

impl Cleaner {
    fn new() -> Self {
        Self {
            template: Regex::new(r"\{\{[^{}]*\}\}").unwrap(),
            media: Regex::new(r"\[\[(?:File|Image|Category|파일|그림|분류):[^\[\]]*\]\]").unwrap(),
            link_with_text: Regex::new(r"\[\[[^\[\]\|]+\|([^\[\]]+)\]\]").unwrap(),
            link_plain: Regex::new(r"\[\[([^\[\]]+)\]\]").unwrap(),
            ref_tag: Regex::new(r"(?s)<ref[^>]*>.*?</ref>|<ref[^/]*/>").unwrap(),
            html_tag: Regex::new(r"<[^>]+>").unwrap(),
            external_link: Regex::new(r"\[https?://\S+(?:\s+[^\]]+)?\]").unwrap(),
            header_line: Regex::new(r"^\s*={2,}\s*(.+?)\s*={2,}\s*$").unwrap(),
            header_strip: Regex::new(r"={2,}\s*([^=\n]+?)\s*={2,}").unwrap(),
            table: Regex::new(r"(?s)\{\|.*?\|\}").unwrap(),
            table_residue: Regex::new(r"(?m)^\s*[|!][!|+\-].*$|^\s*\|\}.*$|^\s*\{\|.*$").unwrap(),
            bold: Regex::new(r"'''([^']+)'''").unwrap(),
            italic: Regex::new(r"''([^']+)''").unwrap(),
            list_marker: Regex::new(r"(?m)^[\*#:;]+\s*").unwrap(),
            blank_lines: Regex::new(r"\n{3,}").unwrap(),
            noise_line: Regex::new(r"^[\s\-=*#:|]*$").unwrap(),
        }
    }

    /// Section headers whose section we should drop entirely (the header
    /// line *and* every line until the next section header at any depth).
    fn is_drop_section(name: &str) -> bool {
        let trimmed = name.trim();
        matches!(
            trimmed,
            "외부 링크"
                | "외부링크"
                | "참고 문헌"
                | "참고문헌"
                | "참고 자료"
                | "참고자료"
                | "각주"
                | "같이 보기"
                | "관련 문서"
                | "관련 항목"
                | "바깥 고리"
                | "바깥고리"
                | "주석"
                | "External links"
                | "References"
                | "See also"
                | "Notes"
                | "Bibliography"
                | "Footnotes"
        )
    }

    fn clean(&self, raw: &str) -> String {
        let mut s = raw.to_string();
        // Remove tables first, then references, then nested templates.
        s = self.table.replace_all(&s, "").into_owned();
        s = self.ref_tag.replace_all(&s, "").into_owned();
        // Templates can nest; iterate until fixed point.
        for _ in 0..8 {
            let after = self.template.replace_all(&s, "").into_owned();
            if after == s {
                break;
            }
            s = after;
        }
        s = self.media.replace_all(&s, "").into_owned();
        s = self.external_link.replace_all(&s, "").into_owned();
        s = self.link_with_text.replace_all(&s, "$1").into_owned();
        s = self.link_plain.replace_all(&s, "$1").into_owned();

        // Section dropper: walk lines, when we see `==Header==` whose name
        // is in the drop-list, set `dropping=true` until the NEXT section
        // header (any level). The Rust `regex` crate has no look-ahead so
        // this is the cleanest way to do scoped drops. Also drop lines
        // that start with `[[파일:` / `[[File:` / `[[Image:` — the
        // body-aware regex sometimes misses long captions, so dropping
        // entire lines is the safer fallback.
        let mut filtered: Vec<&str> = Vec::with_capacity(1024);
        let mut dropping = false;
        for line in s.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("[[파일:")
                || trimmed.starts_with("[[File:")
                || trimmed.starts_with("[[Image:")
                || trimmed.starts_with("[[그림:")
                || trimmed.starts_with("[[분류:")
                || trimmed.starts_with("[[Category:")
            {
                continue;
            }
            if let Some(cap) = self.header_line.captures(line) {
                let name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
                dropping = Self::is_drop_section(name);
                if !dropping {
                    filtered.push(line);
                }
                continue;
            }
            if !dropping {
                filtered.push(line);
            }
        }
        let mut s = filtered.join("\n");

        s = self.header_strip.replace_all(&s, "$1").into_owned();
        s = self.bold.replace_all(&s, "$1").into_owned();
        s = self.italic.replace_all(&s, "$1").into_owned();
        s = self.list_marker.replace_all(&s, "").into_owned();
        s = self.html_tag.replace_all(&s, "").into_owned();
        s = self.table_residue.replace_all(&s, "").into_owned();
        s = self.blank_lines.replace_all(&s, "\n\n").into_owned();
        // Drop empty / punctuation-only lines.
        let kept: Vec<&str> = s
            .lines()
            .filter(|l| !self.noise_line.is_match(l))
            .collect();
        kept.join("\n")
    }
}

fn main() -> anyhow::Result<()> {
    let stdin = io::stdin().lock();
    let stdout = io::stdout().lock();
    let mut out = BufWriter::with_capacity(1 << 16, stdout);
    let mut reader = Reader::from_reader(io::BufReader::with_capacity(1 << 20, stdin));
    let cleaner = Cleaner::new();

    let mut buf = Vec::with_capacity(4096);
    let mut in_text = false;
    let mut in_redirect = false;
    let mut text_buf = String::new();

    let mut articles = 0u64;
    let mut bytes_out = 0u64;

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => {
                eprintln!("xml error: {e}");
                break;
            }
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"text" => {
                    in_text = true;
                    text_buf.clear();
                }
                b"redirect" => in_redirect = true,
                _ => {}
            },
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"redirect" {
                    in_redirect = true;
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"text" => {
                    if in_text && !in_redirect {
                        let cleaned = cleaner.clean(&text_buf);
                        // 800-byte floor filters out stubs / disambiguation /
                        // sport-roster pages (which dominated the previous run
                        // and pinned loss at the wiki-template plateau).
                        if cleaned.len() > 800 {
                            // Articles separated by a double newline + endoftext-like marker.
                            writeln!(out, "{cleaned}\n")?;
                            articles += 1;
                            bytes_out += cleaned.len() as u64;
                            if articles % 1000 == 0 {
                                eprintln!(
                                    "extracted {} articles, {:.1} MB",
                                    articles,
                                    bytes_out as f64 / 1.0e6
                                );
                            }
                        }
                    }
                    in_text = false;
                }
                b"page" => {
                    in_redirect = false;
                }
                _ => {}
            },
            Ok(Event::Text(e)) => {
                if in_text {
                    if let Ok(s) = e.unescape() {
                        text_buf.push_str(&s);
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }

    eprintln!(
        "done: {} articles, {:.1} MB plaintext",
        articles,
        bytes_out as f64 / 1.0e6
    );
    out.flush()?;
    Ok(())
}
