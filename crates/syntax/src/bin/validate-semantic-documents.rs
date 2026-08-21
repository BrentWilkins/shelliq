//! Validate JSON-encoded SemanticDocumentV2 strings and report render/re-lower evidence.

use std::error::Error;
use std::io::{self, BufRead, BufWriter, Write};

use serde::{Deserialize, Serialize};
use shelliq_syntax::semantic::SemanticDocumentV2;

#[derive(Deserialize)]
struct ValidationRequest {
    document: String,
}

#[derive(Serialize)]
struct ValidationReport {
    line: usize,
    valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    rendered: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut stdout = BufWriter::new(io::stdout().lock());
    for (index, line) in stdin.lock().lines().enumerate() {
        let line_number = index + 1;
        let request_line = line?;
        if request_line.trim().is_empty() {
            continue;
        }
        let report = match serde_json::from_str::<ValidationRequest>(&request_line) {
            Ok(request) => match serde_json::from_str::<SemanticDocumentV2>(&request.document) {
                Ok(document) => match document.validate() {
                    Ok(()) => ValidationReport {
                        line: line_number,
                        valid: true,
                        rendered: Some(document.render()),
                        error: None,
                    },
                    Err(error) => ValidationReport {
                        line: line_number,
                        valid: false,
                        rendered: None,
                        error: Some(format!("semantic validation failed: {error}")),
                    },
                },
                Err(error) => ValidationReport {
                    line: line_number,
                    valid: false,
                    rendered: None,
                    error: Some(format!("document JSON decoding failed: {error}")),
                },
            },
            Err(error) => ValidationReport {
                line: line_number,
                valid: false,
                rendered: None,
                error: Some(format!("request JSON decoding failed: {error}")),
            },
        };
        serde_json::to_writer(&mut stdout, &report)?;
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}
