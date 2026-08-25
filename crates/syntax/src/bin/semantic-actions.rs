//! Emit the semantic-action grammar or batch encode/decode JSONL requests.

use std::error::Error;
use std::io::{self, BufRead, BufWriter, Write};

use serde::{Deserialize, Serialize};
use shelliq_syntax::semantic::SemanticDocumentV2;
use shelliq_syntax::semantic::action::SemanticActionV1;

#[derive(Deserialize)]
struct EncodeRequest {
    document: serde_json::Value,
}

#[derive(Deserialize)]
struct DecodeRequest {
    tokens: Vec<u16>,
}

#[derive(Serialize)]
struct ActionReport {
    line: usize,
    valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<u16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rendered: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mode = std::env::args()
        .nth(1)
        .ok_or("usage: semantic-actions manifest|encode|decode")?;
    if mode == "manifest" {
        serde_json::to_writer_pretty(io::stdout().lock(), &SemanticActionV1::grammar_manifest())?;
        println!();
        return Ok(());
    }
    if mode != "encode" && mode != "decode" {
        return Err(format!("unknown semantic-actions mode: {mode}").into());
    }

    let stdin = io::stdin();
    let mut stdout = BufWriter::new(io::stdout().lock());
    for (index, line) in stdin.lock().lines().enumerate() {
        let line_number = index + 1;
        let request = line?;
        if request.trim().is_empty() {
            continue;
        }
        let report = if mode == "encode" {
            encode_line(line_number, &request)
        } else {
            decode_line(line_number, &request)
        };
        serde_json::to_writer(&mut stdout, &report)?;
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}

fn encode_line(line: usize, request: &str) -> ActionReport {
    let result = serde_json::from_str::<EncodeRequest>(request)
        .map_err(|error| format!("request JSON decoding failed: {error}"))
        .and_then(|request| {
            serde_json::from_value::<SemanticDocumentV2>(request.document)
                .map_err(|error| format!("document JSON decoding failed: {error}"))
        })
        .and_then(|document| {
            let tokens = SemanticActionV1::encode(&document).map_err(|error| error.to_string())?;
            Ok((document, tokens))
        });
    match result {
        Ok((document, tokens)) => ActionReport {
            line,
            valid: true,
            tokens: Some(tokens),
            document: serde_json::to_value(document).ok(),
            rendered: None,
            error: None,
        },
        Err(error) => failure(line, error),
    }
}

fn decode_line(line: usize, request: &str) -> ActionReport {
    let result = serde_json::from_str::<DecodeRequest>(request)
        .map_err(|error| format!("request JSON decoding failed: {error}"))
        .and_then(|request| SemanticActionV1::decode(&request.tokens).map_err(|error| error.to_string()));
    match result {
        Ok(document) => ActionReport {
            line,
            valid: true,
            tokens: None,
            document: serde_json::to_value(&document).ok(),
            rendered: Some(document.render()),
            error: None,
        },
        Err(error) => failure(line, error),
    }
}

fn failure(line: usize, error: String) -> ActionReport {
    ActionReport {
        line,
        valid: false,
        tokens: None,
        document: None,
        rendered: None,
        error: Some(error),
    }
}
