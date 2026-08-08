//! Deterministic conversion from curated SFT JSONL into semantic targets.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::SyntaxDocumentV1;
use crate::semantic::{SEMANTIC_SCHEMA_VERSION_V2, SemanticDocumentV2};

/// Version of the semantic-corpus row and manifest envelopes.
pub const CONVERSION_SCHEMA_VERSION: u8 = 1;

/// One prompt and its validated structured target.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCorpusRecord {
    pub conversion_schema_version: u8,
    pub record_id: String,
    pub corpus: String,
    pub source: String,
    pub license: String,
    pub provenance: String,
    pub command: String,
    pub platform: String,
    pub instruction: String,
    pub shell_response: String,
    pub context: String,
    pub semantic_target: SemanticDocumentV2,
}

/// Coverage facts that make every omitted row explicit.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticConversionManifest {
    pub conversion_schema_version: u8,
    pub semantic_schema_version: u8,
    pub input_files: Vec<String>,
    pub total_records: usize,
    pub converted_records: usize,
    pub cst_exempt_record_ids: Vec<String>,
    pub semantic_exempt_record_ids: Vec<String>,
    pub normalized_render_record_ids: Vec<String>,
}

/// In-memory deterministic conversion result.
#[derive(Debug, Eq, PartialEq)]
pub struct SemanticConversion {
    pub records: Vec<SemanticCorpusRecord>,
    pub manifest: SemanticConversionManifest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRecord {
    schema_version: u8,
    record_id: String,
    corpus: String,
    source: String,
    license: String,
    provenance: String,
    command: String,
    platform: String,
    instruction: String,
    response: String,
    context: String,
}

/// Convert every supported row and enforce both exemption files as ratchets.
pub fn convert_curated_corpus(directory: &Path) -> Result<SemanticConversion, Box<dyn Error>> {
    let cst_exemptions = read_exemptions(directory, "cst-exempt.txt")?;
    let semantic_exemptions = read_exemptions(directory, "semantic-exempt.txt")?;
    let overlap: Vec<_> = cst_exemptions.intersection(&semantic_exemptions).collect();
    if !overlap.is_empty() {
        return Err(invalid(format!("CST and semantic exemptions overlap: {overlap:?}")).into());
    }

    let files = corpus_files(directory)?;
    let input_files = files
        .iter()
        .map(|path| file_name(path).map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen_ids = BTreeSet::new();
    let mut records = Vec::new();
    let mut cst_exempt_record_ids = Vec::new();
    let mut semantic_exempt_record_ids = Vec::new();
    let mut normalized_render_record_ids = Vec::new();
    let mut total_records = 0;

    for path in files {
        let contents = fs::read_to_string(&path)?;
        for (index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let source: SourceRecord =
                serde_json::from_str(line).map_err(|error| invalid(format!("{}:{}: {error}", path.display(), index + 1)))?;
            validate_source_record(&source, &path, index + 1)?;
            if !seen_ids.insert(source.record_id.clone()) {
                return Err(invalid(format!("duplicate record_id across corpus: {}", source.record_id)).into());
            }
            total_records += 1;

            let syntax = match SyntaxDocumentV1::parse(&source.response).and_then(|syntax| {
                syntax.validate()?;
                Ok(syntax)
            }) {
                Ok(syntax) => syntax,
                Err(_error) if cst_exemptions.contains(&source.record_id) => {
                    cst_exempt_record_ids.push(source.record_id);
                    continue;
                }
                Err(error) => {
                    return Err(invalid(format!("{} has unexplained CST failure: {error}", source.record_id)).into());
                }
            };
            if cst_exemptions.contains(&source.record_id) {
                return Err(invalid(format!("stale cst-exempt.txt entry now parses: {}", source.record_id)).into());
            }

            let semantic = match SemanticDocumentV2::lower(&syntax).and_then(|semantic| {
                semantic.validate()?;
                Ok(semantic)
            }) {
                Ok(semantic) => semantic,
                Err(_error) if semantic_exemptions.contains(&source.record_id) => {
                    semantic_exempt_record_ids.push(source.record_id);
                    continue;
                }
                Err(error) => {
                    return Err(invalid(format!("{} has unexplained semantic failure: {error}", source.record_id)).into());
                }
            };
            if semantic_exemptions.contains(&source.record_id) {
                return Err(invalid(format!("stale semantic-exempt.txt entry now lowers: {}", source.record_id)).into());
            }

            if semantic.render() != source.response {
                normalized_render_record_ids.push(source.record_id.clone());
            }
            records.push(SemanticCorpusRecord {
                conversion_schema_version: CONVERSION_SCHEMA_VERSION,
                record_id: source.record_id,
                corpus: source.corpus,
                source: source.source,
                license: source.license,
                provenance: source.provenance,
                command: source.command,
                platform: source.platform,
                instruction: source.instruction,
                shell_response: source.response,
                context: source.context,
                semantic_target: semantic,
            });
        }
    }

    require_all_exemptions_seen(&cst_exemptions, &cst_exempt_record_ids, "cst-exempt.txt")?;
    require_all_exemptions_seen(&semantic_exemptions, &semantic_exempt_record_ids, "semantic-exempt.txt")?;
    let manifest = SemanticConversionManifest {
        conversion_schema_version: CONVERSION_SCHEMA_VERSION,
        semantic_schema_version: SEMANTIC_SCHEMA_VERSION_V2,
        input_files,
        total_records,
        converted_records: records.len(),
        cst_exempt_record_ids,
        semantic_exempt_record_ids,
        normalized_render_record_ids,
    };
    Ok(SemanticConversion { records, manifest })
}

fn corpus_files(directory: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = fs::read_dir(directory)?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|path| path.extension().is_some_and(|extension| extension == "jsonl"));
    files.sort();
    if files.is_empty() {
        return Err(invalid(format!("{} contains no JSONL files", directory.display())).into());
    }
    Ok(files)
}

fn read_exemptions(directory: &Path, name: &str) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let contents = fs::read_to_string(directory.join(name))?;
    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect())
}

fn validate_source_record(record: &SourceRecord, path: &Path, line: usize) -> Result<(), io::Error> {
    if record.schema_version != 1 {
        return Err(invalid(format!(
            "{}:{line}: unsupported source schema_version {}",
            path.display(),
            record.schema_version
        )));
    }
    if record.corpus != "distributable" {
        return Err(invalid(format!(
            "{}:{line}: {} record in distributable conversion",
            path.display(),
            record.corpus
        )));
    }
    for (field, value) in [
        ("record_id", record.record_id.as_str()),
        ("source", record.source.as_str()),
        ("license", record.license.as_str()),
        ("provenance", record.provenance.as_str()),
        ("command", record.command.as_str()),
        ("instruction", record.instruction.as_str()),
        ("response", record.response.as_str()),
        ("context", record.context.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(invalid(format!("{}:{line}: {field} must be non-empty", path.display())));
        }
    }
    if !matches!(record.platform.as_str(), "linux" | "darwin") {
        return Err(invalid(format!(
            "{}:{line}: unsupported platform {}",
            path.display(),
            record.platform
        )));
    }
    Ok(())
}

fn require_all_exemptions_seen(expected: &BTreeSet<String>, actual: &[String], name: &str) -> Result<(), io::Error> {
    let actual: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let missing: Vec<&String> = expected
        .iter()
        .filter(|record_id| !actual.contains(record_id.as_str()))
        .collect();
    if !missing.is_empty() {
        return Err(invalid(format!("{name} names missing or stale rows: {missing:?}")));
    }
    Ok(())
}

fn file_name(path: &Path) -> Result<&str, io::Error> {
    path.file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid(format!("non-UTF-8 corpus filename: {}", path.display())))
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
