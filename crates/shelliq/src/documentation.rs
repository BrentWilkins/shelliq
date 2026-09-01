//! Fail-closed compilation of vendored TLDR recipes for the production runtime.
//!
//! There are deliberately no command-family rules here. Documentation prose is
//! ranked generically, option alternatives must exist in the local index, and
//! slots receive only typed values visible in the user's request.

use std::collections::HashSet;

use anyhow::{Context, Result};
use shelliq_harvest::tldr::ParsedExample;
use shelliq_index::{FlagLookup, Index};
use shelliq_syntax::{SyntaxDocumentV1, semantic::SemanticDocumentV2};

use crate::model::Suggestion;

const SLOT_MARKER: char = '\u{1f}';

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompilationStatus {
    Ready,
    NeedsInput,
    NoDocumentation,
}

#[derive(Debug)]
pub struct Compilation {
    pub status: CompilationStatus,
    pub suggestion: Option<Suggestion>,
    pub command_name: Option<String>,
    pub source: Option<String>,
    pub intent: Option<String>,
    pub unresolved_slots: Vec<String>,
    score: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Clarification {
    pub kind: &'static str,
    pub label: String,
    pub question: String,
}

impl Compilation {
    pub fn clarification(&self) -> Option<Clarification> {
        let unresolved = self.unresolved_slots.first()?;
        if unresolved == "ambiguous documentation intent" {
            return Some(Clarification {
                kind: "intent",
                label: "documented operation".into(),
                question: "Which documented operation do you want?".into(),
            });
        }
        if let Some(value) = unresolved.strip_prefix("unused request value ") {
            return Some(Clarification {
                kind: "usage",
                label: value.into(),
                question: format!("How should {value} be used?"),
            });
        }

        let kind = unresolved_slot_kind(unresolved);
        let noun = match kind {
            SlotKind::Path => "file or directory",
            SlotKind::Url => "URL",
            SlotKind::Remote => "remote host or path",
            SlotKind::Integer => "number",
            SlotKind::Text | SlotKind::GenericText => "value",
        };
        Some(Clarification {
            kind: kind.as_str(),
            label: unresolved.clone(),
            question: format!("Which {noun} should be used for `{unresolved}`?"),
        })
    }
}

#[derive(Clone, Debug)]
struct Literal {
    kind: SlotKind,
    value: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotKind {
    Path,
    Url,
    Remote,
    Integer,
    Text,
    GenericText,
}

impl SlotKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Path => "path",
            Self::Url => "url",
            Self::Remote => "remote",
            Self::Integer => "integer",
            Self::Text | Self::GenericText => "text",
        }
    }
}

struct RecipeCompilation {
    command: String,
    unresolved: Vec<String>,
    unused_literals: Vec<String>,
}

pub fn compile(index: &Index, commands: &[String], platform: &str, instruction: &str) -> Result<Compilation> {
    let mut best: Option<Compilation> = None;
    let mut best_documentation_score = 0;
    let mut top_intents = HashSet::new();
    for command in commands {
        if !index.command_exists(command)? {
            continue;
        }
        let examples = shelliq_harvest::tldr::harvest_tldr_family(command, platform)?;
        for (example_index, example) in examples.iter().enumerate() {
            let score = relevance_score(instruction, &example.description);
            if score == 0 {
                continue;
            }
            if score > best_documentation_score {
                best_documentation_score = score;
                best = None;
                top_intents.clear();
            } else if score < best_documentation_score {
                continue;
            }
            top_intents.insert(format!("{command}\n{}", normalize_phrase(&example.description)));
            let Some(mut recipe) = compile_recipe(index, command, instruction, example)? else {
                continue;
            };
            recipe.unresolved.extend(
                recipe
                    .unused_literals
                    .into_iter()
                    .map(|value| format!("unused request value {value}")),
            );
            let source = format!("tldr:{platform}:{command}:{}", example_index + 1);
            let candidate = if recipe.unresolved.is_empty() {
                let Ok(suggestion) = lower_suggestion(&recipe.command) else {
                    continue;
                };
                Compilation {
                    status: CompilationStatus::Ready,
                    suggestion: Some(suggestion),
                    command_name: Some(command.clone()),
                    source: Some(source),
                    intent: Some(example.description.clone()),
                    unresolved_slots: Vec::new(),
                    score,
                }
            } else {
                Compilation {
                    status: CompilationStatus::NeedsInput,
                    suggestion: None,
                    command_name: Some(command.clone()),
                    source: Some(source),
                    intent: Some(example.description.clone()),
                    unresolved_slots: recipe.unresolved,
                    score,
                }
            };
            if is_better(&candidate, best.as_ref()) {
                best = Some(candidate);
            }
        }
    }
    if top_intents.len() > 1 {
        return Ok(Compilation {
            status: CompilationStatus::NeedsInput,
            suggestion: None,
            command_name: None,
            source: None,
            intent: None,
            unresolved_slots: vec!["ambiguous documentation intent".into()],
            score: best_documentation_score,
        });
    }
    Ok(best.unwrap_or(Compilation {
        status: CompilationStatus::NoDocumentation,
        suggestion: None,
        command_name: None,
        source: None,
        intent: None,
        unresolved_slots: Vec::new(),
        score: 0,
    }))
}

pub fn compile_source(index: &Index, platform: &str, instruction: &str, source: &str) -> Result<Compilation> {
    let mut parts = source.split(':');
    let scheme = parts.next();
    let source_platform = parts.next();
    let command = parts.next();
    let ordinal = parts.next();
    if scheme != Some("tldr") || source_platform != Some(platform) || parts.next().is_some() {
        anyhow::bail!("invalid documentation continuation source");
    }
    let command = command
        .filter(|value| !value.is_empty())
        .context("continuation source omitted its command")?;
    let ordinal = ordinal
        .context("continuation source omitted its recipe ordinal")?
        .parse::<usize>()
        .context("continuation recipe ordinal is not an integer")?;
    if ordinal == 0 || !index.command_exists(command)? {
        anyhow::bail!("documentation continuation source is unavailable");
    }
    let examples = shelliq_harvest::tldr::harvest_tldr_family(command, platform)?;
    let example = examples
        .get(ordinal - 1)
        .context("documentation continuation recipe is unavailable")?;
    let mut recipe =
        compile_recipe(index, command, instruction, example)?.context("documentation continuation recipe is unsupported")?;
    recipe.unresolved.extend(
        recipe
            .unused_literals
            .into_iter()
            .map(|value| format!("unused request value {value}")),
    );
    let (status, suggestion) = if recipe.unresolved.is_empty() {
        (CompilationStatus::Ready, Some(lower_suggestion(&recipe.command)?))
    } else {
        (CompilationStatus::NeedsInput, None)
    };
    Ok(Compilation {
        status,
        suggestion,
        command_name: Some(command.to_owned()),
        source: Some(source.to_owned()),
        intent: Some(example.description.clone()),
        unresolved_slots: recipe.unresolved,
        score: 0,
    })
}

fn is_better(candidate: &Compilation, previous: Option<&Compilation>) -> bool {
    let Some(previous) = previous else {
        return true;
    };
    candidate.score > previous.score
        || (candidate.score == previous.score && status_rank(&candidate.status) > status_rank(&previous.status))
        || (candidate.score == previous.score
            && status_rank(&candidate.status) == status_rank(&previous.status)
            && candidate.source < previous.source)
}

fn status_rank(status: &CompilationStatus) -> u8 {
    match status {
        CompilationStatus::Ready => 2,
        CompilationStatus::NeedsInput => 1,
        CompilationStatus::NoDocumentation => 0,
    }
}

fn lower_suggestion(command: &str) -> Result<Suggestion> {
    let syntax = SyntaxDocumentV1::parse(command).context("documentation command is not valid Zsh")?;
    let semantic =
        SemanticDocumentV2::lower(&syntax).context("documentation command is outside the semantic lowering contract")?;
    semantic
        .validate()
        .context("documentation command failed semantic round-trip validation")?;
    let command_name = semantic
        .first_command_name()
        .context("documentation command has no executable")?
        .to_owned();
    Ok(Suggestion {
        command: semantic.render(),
        command_name,
        semantic_json: serde_json::to_string(&semantic).context("serializing documentation semantic document")?,
    })
}

fn compile_recipe(index: &Index, command: &str, instruction: &str, example: &ParsedExample) -> Result<Option<RecipeCompilation>> {
    let Some(marked) = normalize_template(index, command, &example.template)? else {
        return Ok(None);
    };
    if contains_shell_operator(&marked) {
        return Ok(None);
    }
    let literals = request_literals(instruction);
    let mut used = HashSet::new();
    let mut unresolved = Vec::new();
    let mut words = Vec::new();
    for (word_index, raw_word) in marked.split_whitespace().enumerate() {
        let from_placeholder = raw_word.contains(SLOT_MARKER);
        let word = raw_word.replace(SLOT_MARKER, "");
        let bare = strip_shell_quotes(&word);
        if word_index == 0 {
            if bare != command {
                return Ok(None);
            }
            words.push(word);
            continue;
        }
        if let Some(kind) =
            placeholder_kind(bare).or_else(|| (from_placeholder && !bare.starts_with('-')).then(|| unresolved_slot_kind(bare)))
        {
            let match_index = literals
                .iter()
                .enumerate()
                .find_map(|(index, literal)| (!used.contains(&index) && compatible(kind, literal.kind)).then_some(index));
            if let Some(index) = match_index {
                used.insert(index);
                words.push(shell_word(&literals[index].value));
            } else {
                unresolved.push(bare.to_string());
                words.push(word);
            }
            continue;
        }
        if bare.starts_with('-') {
            words.push(word);
            continue;
        }
        if from_placeholder && !instruction_contains(instruction, bare) {
            unresolved.push(bare.to_string());
            words.push(word);
            continue;
        }
        words.push(word);
    }
    unresolved.sort();
    unresolved.dedup();
    let unused_literals = literals
        .iter()
        .enumerate()
        .filter(|(index, _)| !used.contains(index))
        .map(|(_, literal)| literal.value.clone())
        .collect();
    Ok(Some(RecipeCompilation {
        command: words.join(" "),
        unresolved,
        unused_literals,
    }))
}

fn normalize_template(index: &Index, command: &str, raw: &str) -> Result<Option<String>> {
    let mut output = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("{{") {
        output.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            return Ok(None);
        };
        let body = after[..end].trim();
        let body = body
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .unwrap_or(body);
        let alternatives: Vec<&str> = body.split('|').map(str::trim).collect();
        let selected = if alternatives.iter().any(|value| value.starts_with('-')) {
            alternatives.iter().find_map(|alternative| {
                let valid = alternative
                    .split_whitespace()
                    .all(|option| locally_known_option(index, command, option).unwrap_or(false));
                valid.then_some(*alternative)
            })
        } else {
            alternatives.first().copied()
        };
        let Some(selected) = selected else {
            return Ok(None);
        };
        output.push(SLOT_MARKER);
        output.push_str(selected);
        output.push(SLOT_MARKER);
        rest = &after[end + 2..];
    }
    output.push_str(rest);
    Ok(Some(output))
}

fn locally_known_option(index: &Index, command: &str, option: &str) -> Result<bool> {
    if !option.starts_with('-') || option == "-" {
        return Ok(false);
    }
    if matches!(index.lookup_flag(command, option)?, FlagLookup::Exact(_)) {
        return Ok(true);
    }
    if !option.starts_with("--") && option.len() > 2 {
        for character in option[1..].chars() {
            if !matches!(index.lookup_flag(command, &format!("-{character}"))?, FlagLookup::Exact(_)) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    Ok(false)
}

fn relevance_score(instruction: &str, description: &str) -> i64 {
    let requested = normalized_words(instruction);
    let documented = normalized_words(description);
    if documented.is_empty() {
        return 0;
    }
    let overlap = documented.iter().filter(|word| requested.contains(*word)).count() as i64;
    let prefix = normalize_phrase(instruction).starts_with(&normalize_phrase(description));
    if !prefix && overlap * 100 < documented.len() as i64 * 60 {
        return 0;
    }
    overlap * 10 + i64::from(prefix) * 10_000
}

fn normalized_words(value: &str) -> HashSet<String> {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| word.len() > 1)
        .map(|word| stem(word.to_ascii_lowercase()))
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .collect()
}

fn normalize_phrase(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn stem(mut word: String) -> String {
    for suffix in ["ing", "ed", "es", "s"] {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            word.truncate(word.len() - suffix.len());
            break;
        }
    }
    word
}

fn request_literals(instruction: &str) -> Vec<Literal> {
    let mut literals = Vec::new();
    let mut masked = instruction.as_bytes().to_vec();
    let bytes = instruction.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\'' || bytes[index] == b'"' {
            let quote = bytes[index];
            if let Some(relative_end) = bytes[index + 1..].iter().position(|byte| *byte == quote) {
                let end = index + 1 + relative_end;
                let value = &instruction[index + 1..end];
                if !value.is_empty() {
                    literals.push(Literal {
                        kind: SlotKind::Text,
                        value: value.to_string(),
                    });
                }
                masked[index..=end].fill(b' ');
                index = end + 1;
                continue;
            }
        }
        index += 1;
    }
    let unquoted = String::from_utf8(masked).expect("masking quote spans preserves UTF-8");
    for raw in unquoted.split_whitespace() {
        let token = raw.trim_matches(|character: char| matches!(character, ',' | ';' | '(' | ')' | '[' | ']' | '<' | '>'));
        let token = token.trim_end_matches('.');
        if token.is_empty() {
            continue;
        }
        let kind = if token.contains("://") {
            Some(SlotKind::Url)
        } else if token.contains('@') && token.contains(':') {
            Some(SlotKind::Remote)
        } else if is_path_literal(token) {
            Some(SlotKind::Path)
        } else if token.chars().all(|character| character.is_ascii_digit()) {
            Some(SlotKind::Integer)
        } else {
            None
        };
        if let Some(kind) = kind {
            literals.push(Literal {
                kind,
                value: token.to_string(),
            });
        }
    }
    literals
}

fn is_path_literal(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~/")
        || value.rsplit_once('/').map(|(_, leaf)| leaf.contains('.')).unwrap_or_else(|| {
            value
                .rsplit_once('.')
                .is_some_and(|(base, suffix)| !base.is_empty() && (1..=8).contains(&suffix.len()))
        })
}

fn placeholder_kind(value: &str) -> Option<SlotKind> {
    let lowered = value.trim_matches(['\'', '"', '<', '>']).to_ascii_lowercase();
    if lowered.is_empty() || lowered.starts_with('-') {
        return None;
    }
    let parts = || lowered.split(['_', '/', '.', '-']);
    if lowered.contains("://") || parts().any(|part| part == "url") {
        return Some(SlotKind::Url);
    }
    if lowered.contains('@') || parts().any(|part| matches!(part, "host" | "hostname" | "server" | "domain")) {
        return Some(SlotKind::Remote);
    }
    if lowered.contains("path/to/")
        || parts().any(|part| {
            matches!(
                part,
                "archive" | "directory" | "extension" | "file" | "filename" | "package" | "path" | "repository"
            )
        })
    {
        return Some(SlotKind::Path);
    }
    if parts().any(|part| matches!(part, "count" | "id" | "line" | "number" | "pid" | "port" | "time" | "width")) {
        return Some(SlotKind::Integer);
    }
    if matches!(lowered.as_str(), "command" | "name" | "pattern" | "string" | "text")
        || parts().any(|part| {
            matches!(
                part,
                "command"
                    | "date"
                    | "field"
                    | "group"
                    | "key"
                    | "name"
                    | "pattern"
                    | "query"
                    | "regex"
                    | "scope"
                    | "string"
                    | "team"
                    | "text"
                    | "user"
                    | "username"
                    | "value"
            )
        })
    {
        return Some(SlotKind::Text);
    }
    None
}

fn unresolved_slot_kind(value: &str) -> SlotKind {
    placeholder_kind(value)
        .or_else(|| request_literals(value).into_iter().next().map(|literal| literal.kind))
        .unwrap_or(SlotKind::GenericText)
}

fn compatible(slot: SlotKind, literal: SlotKind) -> bool {
    slot == literal
        || (slot == SlotKind::Path && literal == SlotKind::Remote)
        || (slot == SlotKind::GenericText && literal == SlotKind::Text)
}

fn strip_shell_quotes(value: &str) -> &str {
    value
        .strip_prefix('\'')
        .and_then(|item| item.strip_suffix('\''))
        .or_else(|| value.strip_prefix('"').and_then(|item| item.strip_suffix('"')))
        .unwrap_or(value)
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "_./:@%+=,-".contains(character))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn instruction_contains(instruction: &str, value: &str) -> bool {
    let normalized_value = normalize_phrase(value);
    instruction.contains(value)
        || normalize_phrase(instruction)
            .split_whitespace()
            .any(|word| word == normalized_value)
}

fn contains_shell_operator(template: &str) -> bool {
    template.split_whitespace().any(|word| {
        let word = word.replace(SLOT_MARKER, "");
        matches!(word.as_str(), "|" | "||" | "&&" | ";") || word.starts_with('>') || word.starts_with('<')
    })
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "with",
];

#[cfg(test)]
mod tests {
    use super::{
        CompilationStatus, SlotKind, compile, compile_source, normalize_phrase, placeholder_kind, request_literals, shell_word,
        unresolved_slot_kind,
    };
    use shelliq_harvest::{ParsedCommand, ParsedFlag};
    use shelliq_index::Index;

    #[test]
    fn extracts_only_typed_request_literals() {
        let literals = request_literals("Fetch https://example.com/a to /tmp/out.dat with 12 and call it \"daily report\".");
        assert_eq!(literals.len(), 4);
        assert_eq!(literals[0].kind, SlotKind::Text);
        assert_eq!(literals[1].kind, SlotKind::Url);
        assert_eq!(literals[2].kind, SlotKind::Path);
        assert_eq!(literals[3].kind, SlotKind::Integer);
    }

    #[test]
    fn classifies_generic_documentation_slots_without_command_rules() {
        assert_eq!(placeholder_kind("path/to/file.apk"), Some(SlotKind::Path));
        assert_eq!(placeholder_kind("path/to/file.pid"), Some(SlotKind::Path));
        assert_eq!(placeholder_kind("repository_url"), Some(SlotKind::Url));
        assert_eq!(placeholder_kind("port"), Some(SlotKind::Integer));
        assert_eq!(placeholder_kind("playbook"), None);
        assert_eq!(unresolved_slot_kind("8000"), SlotKind::Integer);
        assert_eq!(unresolved_slot_kind("console"), SlotKind::GenericText);
    }

    #[test]
    fn quotes_non_word_request_values() {
        assert_eq!(shell_word("daily report"), "'daily report'");
        assert_eq!(shell_word("/tmp/report.dat"), "/tmp/report.dat");
    }

    #[test]
    fn phrase_normalization_is_punctuation_insensitive() {
        assert_eq!(normalize_phrase("Copy a file."), "copy a file");
    }

    #[test]
    fn status_type_remains_explicit() {
        assert_ne!(CompilationStatus::Ready, CompilationStatus::NeedsInput);
    }

    #[test]
    fn compiles_complete_recipe_and_abstains_on_missing_operand() {
        let mut index = Index::open_in_memory().unwrap();
        index
            .insert_command(
                &shelliq_harvest::resolve_target("grep", false),
                &ParsedCommand {
                    name: "grep".into(),
                    section: "1".into(),
                    platform: "linux".into(),
                    synopsis: String::new(),
                    description: String::new(),
                    flags: vec![ParsedFlag {
                        short: Some("-F".into()),
                        long: Some("--fixed-strings".into()),
                        arg_type: None,
                        arg_required: false,
                        description: "Interpret pattern as a fixed string".into(),
                        group: None,
                        source_line: 1,
                        excerpt: String::new(),
                    }],
                    source_path: "/fixture/grep.1".into(),
                    source_hash: "fixture".into(),
                },
            )
            .unwrap();

        let ready = compile(
            &index,
            &["grep".into()],
            "linux",
            "Search exact string (disables regexes). Use \"needle\" and /tmp/input.dat.",
        )
        .unwrap();
        let ready_source = ready.source.clone().unwrap();
        assert_eq!(ready.status, CompilationStatus::Ready);
        assert_eq!(ready.suggestion.unwrap().command, "grep -F needle /tmp/input.dat");

        let pinned = compile_source(
            &index,
            "linux",
            "Search exact string (disables regexes). Use \"needle\" and /tmp/input.dat.",
            &ready_source,
        )
        .unwrap();
        assert_eq!(pinned.status, CompilationStatus::Ready);
        assert_eq!(pinned.suggestion.unwrap().command, "grep -F needle /tmp/input.dat");
        assert!(compile_source(&index, "linux", "anything", "tldr:linux:grep:0").is_err());

        let incomplete = compile(&index, &["grep".into()], "linux", "Search exact string (disables regexes)").unwrap();
        assert_eq!(incomplete.status, CompilationStatus::NeedsInput);
        assert!(incomplete.suggestion.is_none());
        assert!(!incomplete.unresolved_slots.is_empty());

        let extra = compile(
            &index,
            &["grep".into()],
            "linux",
            "Search exact string (disables regexes). Use \"needle\", /tmp/input.dat, and 42.",
        )
        .unwrap();
        assert_eq!(extra.status, CompilationStatus::NeedsInput);
        assert!(extra.unresolved_slots.iter().any(|slot| slot == "unused request value 42"));
    }
}
