//! Recall@5/MRR relevance measurement for `search_flags` — the "before" (description-only)
//! versus "after" (tldr examples + cross-reference edges, fused by RRF) comparison PLAN.md's
//! P1A acceptance criteria call for, so the gain from tldr ingestion is attributable rather
//! than asserted.
//!
//! Ground truth is the real man pages and vendored tldr pages this project ships, not a hand
//! fixture, so the numbers reflect what a user actually gets on a real machine. Every case
//! whose command has no man page here is skipped, not failed, the same way
//! `crates/harvest/tests/real_pages.rs` stays honest across machines with different
//! packages. BSD-only commands are listed for completeness but always skip on this Linux
//! machine. Help-crawled commands (`ollama`, `kubectl`, `cargo`, `uv`) are not included yet —
//! there is no `--help` crawler to harvest them until that P1A item lands.

use super::*;
use std::process::Command;
use std::time::Instant;

#[allow(dead_code)]
enum Platform {
    Gnu,
    Bsd,
}

struct Case {
    command: &'static str,
    query: &'static str,
    expect: &'static [&'static str],
    platform: Platform,
}

/// Each query is a natural rephrasing of a real vendored tldr bullet for the command, never
/// a copy of the flag spelling itself; `expect` accepts either the short or long spelling,
/// since either counts as the user having found the right flag.
const CASES: &[Case] = &[
    Case {
        command: "curl",
        query: "follow http redirects",
        expect: &["-L", "--location"],
        platform: Platform::Gnu,
    },
    Case {
        command: "curl",
        query: "save the downloaded file using the url's own filename",
        expect: &["-O", "--remote-name"],
        platform: Platform::Gnu,
    },
    Case {
        command: "curl",
        query: "skip certificate validation for a self-signed cert",
        expect: &["-k", "--insecure"],
        platform: Platform::Gnu,
    },
    Case {
        command: "curl",
        query: "send a custom http header",
        expect: &["-H", "--header"],
        platform: Platform::Gnu,
    },
    Case {
        command: "curl",
        query: "use put or delete instead of get",
        expect: &["-X", "--request"],
        platform: Platform::Gnu,
    },
    Case {
        command: "curl",
        query: "route the request through a proxy server",
        expect: &["-x", "--proxy"],
        platform: Platform::Gnu,
    },
    Case {
        command: "grep",
        query: "match a literal string instead of a regex",
        expect: &["-F", "--fixed-strings"],
        platform: Platform::Gnu,
    },
    Case {
        command: "grep",
        query: "show the filename and line number next to each match",
        expect: &["--with-filename", "--line-number"],
        platform: Platform::Gnu,
    },
    Case {
        command: "grep",
        query: "print only the part of the line that matched",
        expect: &["-o", "--only-matching"],
        platform: Platform::Gnu,
    },
    Case {
        command: "grep",
        query: "show lines that do not match the pattern",
        expect: &["-v", "--invert-match"],
        platform: Platform::Gnu,
    },
    Case {
        command: "grep",
        query: "case insensitive search with extended regex syntax",
        expect: &["--extended-regexp", "--ignore-case"],
        platform: Platform::Gnu,
    },
    Case {
        command: "rsync",
        query: "mirror a directory preserving permissions and timestamps",
        expect: &["-a", "--archive"],
        platform: Platform::Gnu,
    },
    Case {
        command: "rsync",
        query: "compress data in transit and show a progress bar",
        expect: &["--compress", "--verbose", "--human-readable", "--partial", "--progress"],
        platform: Platform::Gnu,
    },
    Case {
        command: "rsync",
        query: "only copy files that changed and follow symlinks",
        expect: &["--archive", "--update", "--copy-links"],
        platform: Platform::Gnu,
    },
    Case {
        command: "rsync",
        query: "remove files on the destination that are gone from the source",
        expect: &["-r", "--recursive", "--delete"],
        platform: Platform::Gnu,
    },
    Case {
        command: "ssh",
        query: "connect using a specific private key file",
        expect: &["-i"],
        platform: Platform::Gnu,
    },
    Case {
        command: "ssh",
        query: "reach a server through a jump host",
        expect: &["-J"],
        platform: Platform::Gnu,
    },
    Case {
        command: "chmod",
        query: "change permissions for a whole directory tree at once",
        expect: &["-R", "--recursive"],
        platform: Platform::Gnu,
    },
    Case {
        command: "git-add",
        query: "stage every changed file before committing",
        expect: &["-A", "--all"],
        platform: Platform::Gnu,
    },
    Case {
        command: "git-commit",
        query: "provide the commit message inline instead of an editor",
        expect: &["-m", "--message"],
        platform: Platform::Gnu,
    },
    Case {
        command: "ls",
        query: "show hidden dotfiles too",
        expect: &["-a", "--all"],
        platform: Platform::Gnu,
    },
    Case {
        command: "ls",
        query: "append a symbol showing what kind of entry each file is",
        expect: &["-F", "--classify"],
        platform: Platform::Gnu,
    },
    Case {
        command: "ls",
        query: "list only the directories themselves, not their contents",
        expect: &["-d", "--directory"],
        platform: Platform::Gnu,
    },
];

fn have_page(name: &str) -> bool {
    Command::new("man")
        .args(["-w", name])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn hit_rank(hits: &[FlagRow], expect: &[&str]) -> Option<usize> {
    hits.iter().position(|f| {
        expect
            .iter()
            .any(|e| f.short.as_deref() == Some(*e) || f.long.as_deref() == Some(*e))
    })
}

fn recall_and_mrr(ranks: &[Option<usize>]) -> (f64, f64) {
    let n = ranks.len() as f64;
    let recall = ranks.iter().filter(|r| r.is_some()).count() as f64 / n;
    let mrr = ranks
        .iter()
        .map(|r| r.map(|i| 1.0 / (i as f64 + 1.0)).unwrap_or(0.0))
        .sum::<f64>()
        / n;
    (recall, mrr)
}

/// Records Recall@5 and MRR before (description search alone) and after (tldr examples and
/// cross-reference edges fused by RRF) on real harvested data, printed with `--nocapture` for
/// copying into PLAN.md's P1A acceptance table. Guards against regression rather than
/// asserting a specific gain, since the exact numbers depend on what is installed locally.
#[test]
fn search_relevance_recall_and_mrr_before_and_after_tldr_rrf() {
    let mut idx = Index::open_in_memory().unwrap();
    let mut harvested = std::collections::HashSet::new();
    let mut skipped = 0usize;
    let mut baseline_ranks = Vec::new();
    let mut full_ranks = Vec::new();

    for case in CASES {
        if matches!(case.platform, Platform::Bsd) || !have_page(case.command) {
            skipped += 1;
            continue;
        }
        if harvested.insert(case.command) {
            let target = shelliq_harvest::resolve_target(case.command, false);
            let cmd = shelliq_harvest::harvest(case.command).expect("harvest an installed man page");
            idx.insert_command(&target, &cmd).unwrap();
            idx.insert_tldr_examples(case.command).unwrap();
        }

        baseline_ranks.push(hit_rank(
            &idx.search_flags_description_only(case.command, case.query, 5).unwrap(),
            case.expect,
        ));
        full_ranks.push(hit_rank(&idx.search_flags(case.command, case.query, 5).unwrap(), case.expect));
    }

    let measured = baseline_ranks.len();
    assert!(
        measured >= 10,
        "too few cases ran on this machine to be a meaningful measurement: {measured} of {}",
        CASES.len()
    );

    let (b_recall, b_mrr) = recall_and_mrr(&baseline_ranks);
    let (f_recall, f_mrr) = recall_and_mrr(&full_ranks);

    eprintln!(
        "search relevance ({measured} cases run, {skipped} skipped — no local page): \
         baseline (description only) recall@5={b_recall:.2} mrr={b_mrr:.2} | \
         full (tldr + cross-ref, RRF) recall@5={f_recall:.2} mrr={f_mrr:.2}"
    );

    assert!(
        f_recall >= b_recall && f_mrr >= b_mrr,
        "fused search regressed relative to the description-only baseline: \
         recall@5 {f_recall:.2} < {b_recall:.2} or mrr {f_mrr:.2} < {b_mrr:.2}"
    );
}

/// Latency of `search_flags` with tldr examples, cross-reference expansion, and RRF all in
/// the path — PLAN.md's P1A item, re-measured now that the full pipeline exists. Printed
/// with `--nocapture` for copying into PLAN.md; the assertion is a generous regression guard,
/// not the tighter <10ms target quoted for fuzzy search in section 3.
#[test]
fn search_flags_latency_with_full_pipeline() {
    let mut idx = Index::open_in_memory().unwrap();
    for case in CASES {
        if matches!(case.platform, Platform::Bsd) || !have_page(case.command) {
            continue;
        }
        let target = shelliq_harvest::resolve_target(case.command, false);
        if idx.command_exists(case.command).unwrap_or(false) {
            continue;
        }
        let Ok(cmd) = shelliq_harvest::harvest(case.command) else {
            continue;
        };
        idx.insert_command(&target, &cmd).unwrap();
        idx.insert_tldr_examples(case.command).unwrap();
    }

    let runnable: Vec<&Case> = CASES
        .iter()
        .filter(|c| idx.command_exists(c.command).unwrap_or(false))
        .collect();
    assert!(
        runnable.len() >= 10,
        "too few cases ran on this machine to measure latency: {}",
        runnable.len()
    );

    let mut micros: Vec<u128> = Vec::new();
    for _ in 0..20 {
        for case in &runnable {
            let start = Instant::now();
            idx.search_flags(case.command, case.query, 5).unwrap();
            micros.push(start.elapsed().as_micros());
        }
    }
    micros.sort_unstable();
    let p50 = micros[micros.len() / 2];
    let p95 = micros[micros.len() * 95 / 100];

    eprintln!(
        "search_flags latency, full pipeline, {} samples: p50={p50}us p95={p95}us",
        micros.len()
    );

    assert!(p95 < 25_000, "search_flags p95 latency regressed badly: {p95}us");
}
