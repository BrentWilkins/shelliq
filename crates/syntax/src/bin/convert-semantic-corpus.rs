//! Convert curated schema-v1 JSONL rows into validated semantic targets.

use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use shelliq_syntax::corpus::convert_curated_corpus;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let directory = required_path(arguments.next(), "CORPUS_DIRECTORY")?;
    let output = required_path(arguments.next(), "OUTPUT.jsonl")?;
    let manifest = required_path(arguments.next(), "MANIFEST.json")?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }
    if output == manifest {
        return Err("OUTPUT.jsonl and MANIFEST.json must be different paths".into());
    }
    require_new(&output)?;
    require_new(&manifest)?;

    let conversion = convert_curated_corpus(&directory)?;
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
        "converted {}/{} records ({} CST exemptions, {} semantic exemptions, {} normalized renders)",
        conversion.manifest.converted_records,
        conversion.manifest.total_records,
        conversion.manifest.cst_exempt_record_ids.len(),
        conversion.manifest.semantic_exempt_record_ids.len(),
        conversion.manifest.normalized_render_record_ids.len(),
    );
    println!("output: {}", output.display());
    println!("manifest: {}", manifest.display());
    Ok(())
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name}; {}", usage()).into())
}

fn usage() -> &'static str {
    "usage: convert-semantic-corpus CORPUS_DIRECTORY OUTPUT.jsonl MANIFEST.json"
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
