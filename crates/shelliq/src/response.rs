use anyhow::{Context, Result};
use serde::Serialize;

use crate::{documentation::Clarification, model::Suggestion};

const RESPONSE_VERSION: u8 = 1;

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum SuggestResponse<'a> {
    Ready {
        v: u8,
        command: &'a str,
        semantic: serde_json::Value,
        source: &'a str,
    },
    NeedsInput {
        v: u8,
        clarification: ClarificationResponse<'a>,
    },
    NoDocumentation {
        v: u8,
        message: &'a str,
    },
}

#[derive(Debug, Serialize)]
struct ClarificationResponse<'a> {
    kind: &'a str,
    label: &'a str,
    question: &'a str,
}

pub fn ready(suggestion: &Suggestion, source: &str) -> Result<String> {
    let semantic = serde_json::from_str(&suggestion.semantic_json).context("parsing validated semantic document for response")?;
    serialize(&SuggestResponse::Ready {
        v: RESPONSE_VERSION,
        command: &suggestion.command,
        semantic,
        source,
    })
}

pub fn needs_input(clarification: &Clarification) -> Result<String> {
    serialize(&SuggestResponse::NeedsInput {
        v: RESPONSE_VERSION,
        clarification: ClarificationResponse {
            kind: clarification.kind,
            label: &clarification.label,
            question: &clarification.question,
        },
    })
}

pub fn no_documentation(message: &str) -> Result<String> {
    serialize(&SuggestResponse::NoDocumentation {
        v: RESPONSE_VERSION,
        message,
    })
}

fn serialize(response: &SuggestResponse<'_>) -> Result<String> {
    serde_json::to_string(response).context("serializing suggestion response")
}

#[cfg(test)]
mod tests {
    use super::{needs_input, no_documentation};
    use crate::documentation::Clarification;

    #[test]
    fn needs_input_contains_no_command_field() {
        let response = needs_input(&Clarification {
            kind: "path",
            label: "file".into(),
            question: "Which file should be used?".into(),
        })
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["v"], 1);
        assert_eq!(value["status"], "needs_input");
        assert!(value.get("command").is_none());
        assert!(value.get("semantic").is_none());
    }

    #[test]
    fn no_documentation_contains_no_command_field() {
        let response = no_documentation("no recipe").unwrap();
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["status"], "no_documentation");
        assert!(value.get("command").is_none());
    }
}
