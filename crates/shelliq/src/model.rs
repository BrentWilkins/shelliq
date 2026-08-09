//! Guarded loopback model boundary for experimental command suggestions.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use shelliq_syntax::semantic::SemanticDocumentV2;

const CHAT_PATH: &str = "/v1/chat/completions";
const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;
const MAX_GENERATION_TOKENS: u16 = 192;

#[derive(Debug, Eq, PartialEq)]
pub struct Suggestion {
    pub command: String,
    pub semantic_json: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    messages: [ChatMessage<'a>; 1],
    temperature: u8,
    max_tokens: u16,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

pub fn validate_endpoint(endpoint: &str) -> Result<()> {
    let remainder = ["http://127.0.0.1", "http://[::1]"]
        .into_iter()
        .find_map(|prefix| endpoint.strip_prefix(prefix))
        .context("model endpoint must use loopback HTTP")?;

    let port = if remainder == CHAT_PATH {
        None
    } else {
        let port = remainder
            .strip_prefix(':')
            .and_then(|value| value.strip_suffix(CHAT_PATH))
            .context("model endpoint must end with /v1/chat/completions")?;
        Some(port.parse::<u16>().context("model endpoint has an invalid port")?)
    };
    if port == Some(0) {
        bail!("model endpoint port must be nonzero");
    }
    Ok(())
}

pub fn suggest(endpoint: &str, platform: &str, context: &str, instruction: &str, timeout: Duration) -> Result<Suggestion> {
    validate_endpoint(endpoint)?;
    if instruction.trim().is_empty() {
        bail!("suggestion instruction must not be empty");
    }
    let escaped_context = context.replace("</context>", "&lt;/context&gt;");
    let user_message = format!(
        "# platform: {platform}\n<context>\n{escaped_context}\n</context>\n\n{}",
        instruction.trim()
    );
    let request = ChatRequest {
        messages: [ChatMessage {
            role: "user",
            content: &user_message,
        }],
        temperature: 0,
        max_tokens: MAX_GENERATION_TOKENS,
        stream: false,
    };
    let config = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .timeout_global(Some(timeout))
        .build();
    let agent: ureq::Agent = config.into();
    let mut response = agent
        .post(endpoint)
        .send_json(&request)
        .context("model server request failed")?;
    if response.status().as_u16() != 200 {
        bail!("model server returned HTTP {}", response.status().as_u16());
    }
    let body = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_vec()
        .context("model server response was invalid or too large")?;
    let response: ChatResponse = serde_json::from_slice(&body).context("model server did not return a chat completion")?;
    let content = response
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .filter(|content| !content.is_empty())
        .context("model server returned no assistant content")?;
    validate_generation(content)
}

fn validate_generation(content: String) -> Result<Suggestion> {
    let semantic: SemanticDocumentV2 = serde_json::from_str(&content).context("model output is not SemanticDocumentV2 JSON")?;
    semantic
        .validate()
        .context("model output failed semantic render/reparse validation")?;
    Ok(Suggestion {
        command: semantic.render(),
        semantic_json: content,
    })
}

#[cfg(test)]
mod tests {
    use super::{validate_endpoint, validate_generation};

    #[test]
    fn accepts_only_loopback_chat_completion_endpoints() {
        for endpoint in [
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://[::1]:8080/v1/chat/completions",
            "http://127.0.0.1/v1/chat/completions",
        ] {
            validate_endpoint(endpoint).unwrap();
        }
    }

    #[test]
    fn rejects_expanded_endpoint_trust_boundaries() {
        for endpoint in [
            "https://127.0.0.1:8080/v1/chat/completions",
            "http://example.com/v1/chat/completions",
            "http://localhost:11434/v1/chat/completions",
            "http://user@127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:8080/health",
            "http://127.0.0.1:0/v1/chat/completions",
            "http://127.0.0.1:8080/v1/chat/completions?token=x",
        ] {
            assert!(validate_endpoint(endpoint).is_err(), "accepted {endpoint}");
        }
    }

    #[test]
    fn renders_only_round_trip_valid_semantic_documents() {
        let semantic = String::from(
            r#"{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"cp"},"a":[{"s":"-a"},{"s":"src/"},{"s":"dest/"}]}]}]}"#,
        );
        let result = validate_generation(semantic.clone()).unwrap();

        assert_eq!(result.command, "cp -a src/ dest/");
        assert_eq!(result.semantic_json, semantic);
    }

    #[test]
    fn rejects_semantic_documents_that_fail_round_trip_validation() {
        let error = validate_generation(String::from(r#"{"v":1,"d":"zsh","s":[]}"#)).unwrap_err();

        assert!(error.to_string().contains("semantic render/reparse"));
    }
}
