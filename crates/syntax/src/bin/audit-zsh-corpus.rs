//! Compare the structural Zsh parser with native `zsh -n` over JSONL records.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use shelliq_syntax::SyntaxDocumentV1;

const SAMPLE_LIMIT: usize = 20;

#[derive(Deserialize)]
struct Record {
    record_id: String,
    response: String,
}

#[derive(Debug, Default, Serialize)]
struct AuditReport {
    records: usize,
    native_zsh_valid: usize,
    syntax_tree_valid: usize,
    parsers_agree: usize,
    native_only: usize,
    syntax_tree_only: usize,
    both_reject: usize,
    native_only_record_ids: Vec<String>,
    syntax_tree_only_record_ids: Vec<String>,
    both_reject_record_ids: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let dataset = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: audit-zsh-corpus DATASET.jsonl [ZSH]")?;
    let zsh = arguments.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("zsh"));
    if arguments.next().is_some() {
        return Err("usage: audit-zsh-corpus DATASET.jsonl [ZSH]".into());
    }

    let report = audit(&dataset, &zsh)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn audit(dataset: &Path, zsh: &Path) -> Result<AuditReport, Box<dyn std::error::Error>> {
    let mut report = AuditReport::default();
    for (index, line) in BufReader::new(File::open(dataset)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record =
            serde_json::from_str(&line).map_err(|error| format!("{}:{}: {error}", dataset.display(), index + 1))?;
        report.records += 1;

        let native_valid = native_zsh_accepts(zsh, &record.response)?;
        let tree_valid = SyntaxDocumentV1::parse(&record.response).is_ok();
        report.native_zsh_valid += usize::from(native_valid);
        report.syntax_tree_valid += usize::from(tree_valid);

        match (native_valid, tree_valid) {
            (true, true) => report.parsers_agree += 1,
            (false, false) => {
                report.parsers_agree += 1;
                report.both_reject += 1;
                push_sample(&mut report.both_reject_record_ids, record.record_id);
            }
            (true, false) => {
                report.native_only += 1;
                push_sample(&mut report.native_only_record_ids, record.record_id);
            }
            (false, true) => {
                report.syntax_tree_only += 1;
                push_sample(&mut report.syntax_tree_only_record_ids, record.record_id);
            }
        }
    }
    Ok(report)
}

fn native_zsh_accepts(zsh: &Path, source: &str) -> Result<bool, std::io::Error> {
    Command::new(zsh)
        .args(["-f", "-n", "-c", source])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
}

fn push_sample(samples: &mut Vec<String>, record_id: String) {
    if samples.len() < SAMPLE_LIMIT {
        samples.push(record_id);
    }
}
