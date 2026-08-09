//! Convert an audited schema-v1 JSONL dataset into validated semantic targets.

use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use shelliq_syntax::SyntaxDocumentV1;
use shelliq_syntax::corpus::{CONVERSION_SCHEMA_VERSION, SemanticCorpusRecord};
use shelliq_syntax::semantic::{SEMANTIC_SCHEMA_VERSION_V2, SemanticDocumentV2};

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

#[derive(Serialize)]
struct DatasetConversionManifest {
    conversion_schema_version: u8,
    semantic_schema_version: u8,
    input_file: String,
    total_records: usize,
    converted_records: usize,
    cst_rejected_record_ids: Vec<String>,
    semantic_rejected_record_ids: Vec<String>,
    normalized_render_record_ids: Vec<String>,
}

struct DatasetConversion {
    records: Vec<SemanticCorpusRecord>,
    manifest: DatasetConversionManifest,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = required_path(arguments.next(), "INPUT.jsonl")?;
    let output = required_path(arguments.next(), "OUTPUT.jsonl")?;
    let manifest = required_path(arguments.next(), "MANIFEST.json")?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }
    if output == manifest || input == output || input == manifest {
        return Err("input, output, and manifest paths must be different".into());
    }
    if !input.is_file() {
        return Err(format!("input does not exist: {}", input.display()).into());
    }
    require_new(&output)?;
    require_new(&manifest)?;

    let contents = fs::read_to_string(&input)?;
    let conversion = convert_dataset(&contents, &input)?;
    create_parent(&output)?;
    create_parent(&manifest)?;

    let mut writer = BufWriter::new(File::create(&output)?);
    for record in &conversion.records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    fs::write(&manifest, serde_json::to_string_pretty(&conversion.manifest)? + "\n")?;

    println!(
        "converted {}/{} records ({} CST rejects, {} semantic rejects, {} normalized renders)",
        conversion.manifest.converted_records,
        conversion.manifest.total_records,
        conversion.manifest.cst_rejected_record_ids.len(),
        conversion.manifest.semantic_rejected_record_ids.len(),
        conversion.manifest.normalized_render_record_ids.len(),
    );
    println!("output: {}", output.display());
    println!("manifest: {}", manifest.display());
    Ok(())
}

fn convert_dataset(contents: &str, input: &Path) -> Result<DatasetConversion, Box<dyn Error>> {
    let mut records = Vec::new();
    let mut seen_ids = BTreeSet::new();
    let mut cst_rejected_record_ids = Vec::new();
    let mut semantic_rejected_record_ids = Vec::new();
    let mut normalized_render_record_ids = Vec::new();
    let mut total_records = 0;

    for (index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        total_records += 1;
        let source: SourceRecord = serde_json::from_str(line)
            .map_err(|error| format!("{}:{}: invalid source row: {error}", input.display(), index + 1))?;
        validate_source(&source, input, index + 1)?;
        if !seen_ids.insert(source.record_id.clone()) {
            return Err(format!("duplicate record_id: {}", source.record_id).into());
        }

        let syntax = match SyntaxDocumentV1::parse(&source.response).and_then(|document| {
            document.validate()?;
            Ok(document)
        }) {
            Ok(syntax) => syntax,
            Err(_) => {
                cst_rejected_record_ids.push(source.record_id);
                continue;
            }
        };
        let semantic = match SemanticDocumentV2::lower(&syntax).and_then(|document| {
            document.validate()?;
            Ok(document)
        }) {
            Ok(semantic) => semantic,
            Err(_) => {
                semantic_rejected_record_ids.push(source.record_id);
                continue;
            }
        };
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

    Ok(DatasetConversion {
        manifest: DatasetConversionManifest {
            conversion_schema_version: CONVERSION_SCHEMA_VERSION,
            semantic_schema_version: SEMANTIC_SCHEMA_VERSION_V2,
            input_file: input.display().to_string(),
            total_records,
            converted_records: records.len(),
            cst_rejected_record_ids,
            semantic_rejected_record_ids,
            normalized_render_record_ids,
        },
        records,
    })
}

fn validate_source(record: &SourceRecord, path: &Path, line: usize) -> Result<(), Box<dyn Error>> {
    if record.schema_version != 1 {
        return Err(format!(
            "{}:{line}: unsupported schema_version {}",
            path.display(),
            record.schema_version
        )
        .into());
    }
    if record.corpus != "distributable" {
        return Err(format!(
            "{}:{line}: corpus must be distributable, got {:?}",
            path.display(),
            record.corpus
        )
        .into());
    }
    for (name, value) in [
        ("record_id", record.record_id.as_str()),
        ("source", record.source.as_str()),
        ("license", record.license.as_str()),
        ("provenance", record.provenance.as_str()),
        ("command", record.command.as_str()),
        ("instruction", record.instruction.as_str()),
        ("response", record.response.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{}:{line}: {name} must be non-empty", path.display()).into());
        }
    }
    if !matches!(record.platform.as_str(), "linux" | "darwin") {
        return Err(format!("{}:{line}: unsupported platform {:?}", path.display(), record.platform).into());
    }
    Ok(())
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value.map(PathBuf::from).ok_or_else(|| usage_for(name).into())
}

fn require_new(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("output already exists: {}", path.display()).into());
    }
    Ok(())
}

fn create_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn usage_for(_name: &str) -> String {
    usage()
}

fn usage() -> String {
    "usage: convert-semantic-dataset INPUT.jsonl OUTPUT.jsonl MANIFEST.json".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorizes_invalid_syntax_without_losing_valid_rows() {
        let contents = concat!(
            r#"{"schema_version":1,"record_id":"ok","corpus":"distributable","source":"test","license":"MIT","provenance":"test","command":"printf","platform":"linux","instruction":"Print text","response":"printf hi","context":""}"#,
            "\n",
            r#"{"schema_version":1,"record_id":"bad","corpus":"distributable","source":"test","license":"MIT","provenance":"test","command":"printf","platform":"linux","instruction":"Print text","response":"printf '","context":""}"#,
            "\n",
        );

        let conversion = convert_dataset(contents, Path::new("input.jsonl")).unwrap();

        assert_eq!(conversion.records.len(), 1);
        assert_eq!(conversion.manifest.total_records, 2);
        assert_eq!(conversion.manifest.cst_rejected_record_ids, ["bad"]);
        assert!(conversion.manifest.semantic_rejected_record_ids.is_empty());
    }
}
