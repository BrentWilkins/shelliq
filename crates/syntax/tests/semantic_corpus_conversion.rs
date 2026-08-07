use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use shelliq_syntax::corpus::convert_curated_corpus;

fn corpus_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/corpus")
}

#[test]
fn real_curated_conversion_is_deterministic_and_lossless() {
    let first = convert_curated_corpus(&corpus_directory()).expect("curated corpus converts");
    let second = convert_curated_corpus(&corpus_directory()).expect("repeat conversion succeeds");

    assert_eq!(first, second);
    assert_eq!(first.manifest.total_records, 626);
    assert_eq!(first.manifest.converted_records, 617);
    assert_eq!(first.manifest.cst_exempt_record_ids.len(), 6);
    assert_eq!(first.manifest.semantic_exempt_record_ids.len(), 3);
    assert_eq!(first.manifest.input_files.len(), 13);
    assert!(
        first
            .manifest
            .normalized_render_record_ids
            .iter()
            .any(|record_id| record_id == "curated:archives-and-encoding:linux:base64-decode")
    );
    for record in first.records {
        let encoded = serde_json::to_string(&record).unwrap();
        let decoded = serde_json::from_str(&encoded).unwrap();
        assert_eq!(record, decoded);
        record.semantic_target.validate().unwrap();
    }
}

#[test]
fn conversion_rejects_unexplained_and_stale_exemptions() {
    let fixture = FixtureDirectory::new();
    fixture.write_exemptions("", "");
    fixture.write_records(&[
        source_record("fixture:valid", "print ok"),
        source_record("fixture:declaration", "typeset -i count=0"),
    ]);

    let error = convert_curated_corpus(&fixture.path).unwrap_err();
    assert!(error.to_string().contains("unexplained semantic failure"));

    fixture.write_exemptions("", "fixture:declaration\n");
    let conversion = convert_curated_corpus(&fixture.path).expect("explained failure converts");
    assert_eq!(conversion.manifest.converted_records, 1);
    assert_eq!(conversion.manifest.semantic_exempt_record_ids, ["fixture:declaration"]);

    fixture.write_exemptions("", "fixture:valid\nfixture:declaration\n");
    let error = convert_curated_corpus(&fixture.path).unwrap_err();
    assert!(error.to_string().contains("stale semantic-exempt.txt entry"));
}

fn source_record(record_id: &str, response: &str) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "record_id": record_id,
        "corpus": "distributable",
        "source": "fixture",
        "license": "MIT OR Apache-2.0",
        "provenance": "fixture:test",
        "command": response.split_whitespace().next().unwrap(),
        "platform": "linux",
        "instruction": "Run the fixture.",
        "response": response,
        "context": "fixture: test context."
    })
}

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn new() -> Self {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("shelliq-semantic-conversion-{}-{unique}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn write_exemptions(&self, cst: &str, semantic: &str) {
        fs::write(self.path.join("cst-exempt.txt"), cst).unwrap();
        fs::write(self.path.join("semantic-exempt.txt"), semantic).unwrap();
    }

    fn write_records(&self, records: &[serde_json::Value]) {
        let mut contents = records
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join("\n");
        contents.push('\n');
        fs::write(self.path.join("fixture.jsonl"), contents).unwrap();
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).unwrap();
    }
}
