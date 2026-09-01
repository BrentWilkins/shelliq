//! Validate the checked-in curated corpus against the CST and semantic contracts.
//!
//! Every response is expected to parse losslessly into a CST and then lower into
//! the semantic AST and round-trip. Rows that the current implementation cannot
//! handle are listed in one of two exemption files rather than dropped from the
//! corpus, because the shapes they teach are a large part of why the curated
//! corpus exists:
//!
//! - `cst-exempt.txt` — native `zsh -n` accepts the response but the structural
//!   parser rejects it. These are the `native_only` rows in audit-zsh-corpus
//!   terms, mostly Zsh glob qualifiers.
//! - `semantic-exempt.txt` — the CST parses but the semantic slice cannot lower
//!   it. The slice currently covers only `command`, `redirected_statement`, and
//!   `pipeline` (see `lower_statement` in `src/semantic.rs`).
//!
//! Both lists are ratchets, not escape hatches: each asserts that its entries
//! actually still fail. When the parser or the semantic slice grows to cover a
//! shape, the stale exemptions fail this test until they are deleted.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use shelliq_syntax::{SyntaxDocumentV1, corpus::is_supervised_corpus_file, semantic::SemanticDocumentV2};

#[derive(Deserialize)]
struct CuratedRecord {
    record_id: String,
    response: String,
}

fn corpus_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/corpus")
}

fn read_exemptions(directory: &Path, name: &str) -> BTreeSet<String> {
    let path = directory.join(name);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

fn read_records(directory: &Path) -> Vec<CuratedRecord> {
    let mut files: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| is_supervised_corpus_file(path))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "curated corpus must contain at least one JSONL file");

    let mut records = Vec::new();
    for file in files {
        let contents = fs::read_to_string(&file).unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        for (index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: CuratedRecord =
                serde_json::from_str(line).unwrap_or_else(|error| panic!("{}:{}: {error}", file.display(), index + 1));
            records.push(record);
        }
    }
    records
}

fn assert_known(exemptions: &BTreeSet<String>, records: &[CuratedRecord], name: &str) {
    let known: BTreeSet<&str> = records.iter().map(|record| record.record_id.as_str()).collect();
    let unknown: Vec<&String> = exemptions
        .iter()
        .filter(|exemption| !known.contains(exemption.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "{name} lists record_ids that are not in the corpus: {unknown:#?}"
    );
}

#[test]
fn cst_exemptions_are_a_ratchet() {
    let directory = corpus_directory();
    let exemptions = read_exemptions(&directory, "cst-exempt.txt");
    let records = read_records(&directory);
    assert_known(&exemptions, &records, "cst-exempt.txt");

    let mut newly_supported = Vec::new();
    let mut missing_exemption = Vec::new();
    for record in &records {
        let parsed = SyntaxDocumentV1::parse(&record.response)
            .map(|syntax| syntax.render() == record.response && syntax.validate().is_ok());
        let accepted = matches!(parsed, Ok(true));
        match (accepted, exemptions.contains(&record.record_id)) {
            (true, false) | (false, true) => {}
            (false, false) => missing_exemption.push(format!("{} :: {}", record.record_id, record.response)),
            (true, true) => newly_supported.push(record.record_id.clone()),
        }
    }

    assert!(
        missing_exemption.is_empty(),
        "these rows fail lossless CST parsing and are not in cst-exempt.txt: {missing_exemption:#?}"
    );
    assert!(
        newly_supported.is_empty(),
        "the parser now accepts these rows; delete them from cst-exempt.txt: {newly_supported:#?}"
    );
}

#[test]
fn semantic_exemptions_are_a_ratchet() {
    let directory = corpus_directory();
    let cst_exemptions = read_exemptions(&directory, "cst-exempt.txt");
    let exemptions = read_exemptions(&directory, "semantic-exempt.txt");
    let records = read_records(&directory);
    assert_known(&exemptions, &records, "semantic-exempt.txt");

    let overlap: Vec<&String> = exemptions.intersection(&cst_exemptions).collect();
    assert!(
        overlap.is_empty(),
        "rows that cannot parse cannot lower either; list them only in cst-exempt.txt: {overlap:#?}"
    );

    let mut newly_supported = Vec::new();
    let mut missing_exemption = Vec::new();
    for record in &records {
        if cst_exemptions.contains(&record.record_id) {
            continue;
        }
        let Ok(syntax) = SyntaxDocumentV1::parse(&record.response) else {
            continue;
        };
        let lowered = SemanticDocumentV2::lower(&syntax).and_then(|semantic| semantic.validate());
        match (lowered.is_ok(), exemptions.contains(&record.record_id)) {
            (true, false) | (false, true) => {}
            (false, false) => missing_exemption.push(format!("{} :: {}", record.record_id, record.response)),
            (true, true) => newly_supported.push(record.record_id.clone()),
        }
    }

    assert!(
        missing_exemption.is_empty(),
        "these rows fail semantic lowering and are not in semantic-exempt.txt: {missing_exemption:#?}"
    );
    assert!(
        newly_supported.is_empty(),
        "the semantic slice now covers these rows; delete them from semantic-exempt.txt: {newly_supported:#?}"
    );
}
