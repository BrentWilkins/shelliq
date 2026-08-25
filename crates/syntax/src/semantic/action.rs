//! Lossless, grammar-constrained action representation for semantic documents.

use std::fmt;

use serde::Serialize;

use super::{
    AssignmentV2, CommandV2, PipelineOperatorV1, RedirectOperatorV2, RedirectV2, SEMANTIC_SCHEMA_VERSION_V2, SemanticDocumentV2,
    StatementV2, WordV1,
};
use crate::ShellDialect;

pub const ACTION_SCHEMA_VERSION: u8 = 1;
pub const ACTION_VOCAB_SIZE: u16 = 320;
pub const BYTE_OFFSET: u16 = 64;

pub mod token {
    pub const PAD: u16 = 0;
    pub const BOS: u16 = 1;
    pub const EOS: u16 = 2;
    pub const DOC_START: u16 = 3;
    pub const DOC_END: u16 = 4;
    pub const PIPELINE_START: u16 = 5;
    pub const PIPELINE_END: u16 = 6;
    pub const COMMAND_START: u16 = 7;
    pub const COMMAND_END: u16 = 8;
    pub const COMMAND_NAME_START: u16 = 9;
    pub const ARGUMENT_START: u16 = 10;
    pub const REDIRECT_START: u16 = 11;
    pub const REDIRECT_END: u16 = 12;
    pub const REDIRECT_DESCRIPTOR_START: u16 = 13;
    pub const REDIRECT_TARGET_START: u16 = 14;
    pub const DECLARATION_START: u16 = 15;
    pub const DECLARATION_END: u16 = 16;
    pub const UTILITY_START: u16 = 17;
    pub const OPTION_START: u16 = 18;
    pub const ASSIGNMENT_START: u16 = 19;
    pub const ASSIGNMENT_END: u16 = 20;
    pub const ASSIGNMENT_NAME_START: u16 = 21;
    pub const ASSIGNMENT_VALUE_START: u16 = 22;
    pub const WORD_END: u16 = 23;
    pub const PIPE_STDOUT: u16 = 24;
    pub const PIPE_STDOUT_STDERR: u16 = 25;
    pub const REDIRECT_READ: u16 = 26;
    pub const REDIRECT_WRITE: u16 = 27;
    pub const REDIRECT_APPEND: u16 = 28;
    pub const REDIRECT_CLOBBER: u16 = 29;
    pub const REDIRECT_READ_WRITE: u16 = 30;
    pub const REDIRECT_DUPLICATE_INPUT: u16 = 31;
    pub const REDIRECT_DUPLICATE_OUTPUT: u16 = 32;
    pub const REDIRECT_HERE_STRING: u16 = 33;
}

const TOKEN_NAMES: &[(u16, &str)] = &[
    (token::PAD, "PAD"),
    (token::BOS, "BOS"),
    (token::EOS, "EOS"),
    (token::DOC_START, "DOC_START"),
    (token::DOC_END, "DOC_END"),
    (token::PIPELINE_START, "PIPELINE_START"),
    (token::PIPELINE_END, "PIPELINE_END"),
    (token::COMMAND_START, "COMMAND_START"),
    (token::COMMAND_END, "COMMAND_END"),
    (token::COMMAND_NAME_START, "COMMAND_NAME_START"),
    (token::ARGUMENT_START, "ARGUMENT_START"),
    (token::REDIRECT_START, "REDIRECT_START"),
    (token::REDIRECT_END, "REDIRECT_END"),
    (token::REDIRECT_DESCRIPTOR_START, "REDIRECT_DESCRIPTOR_START"),
    (token::REDIRECT_TARGET_START, "REDIRECT_TARGET_START"),
    (token::DECLARATION_START, "DECLARATION_START"),
    (token::DECLARATION_END, "DECLARATION_END"),
    (token::UTILITY_START, "UTILITY_START"),
    (token::OPTION_START, "OPTION_START"),
    (token::ASSIGNMENT_START, "ASSIGNMENT_START"),
    (token::ASSIGNMENT_END, "ASSIGNMENT_END"),
    (token::ASSIGNMENT_NAME_START, "ASSIGNMENT_NAME_START"),
    (token::ASSIGNMENT_VALUE_START, "ASSIGNMENT_VALUE_START"),
    (token::WORD_END, "WORD_END"),
    (token::PIPE_STDOUT, "PIPE_STDOUT"),
    (token::PIPE_STDOUT_STDERR, "PIPE_STDOUT_STDERR"),
    (token::REDIRECT_READ, "REDIRECT_READ"),
    (token::REDIRECT_WRITE, "REDIRECT_WRITE"),
    (token::REDIRECT_APPEND, "REDIRECT_APPEND"),
    (token::REDIRECT_CLOBBER, "REDIRECT_CLOBBER"),
    (token::REDIRECT_READ_WRITE, "REDIRECT_READ_WRITE"),
    (token::REDIRECT_DUPLICATE_INPUT, "REDIRECT_DUPLICATE_INPUT"),
    (token::REDIRECT_DUPLICATE_OUTPUT, "REDIRECT_DUPLICATE_OUTPUT"),
    (token::REDIRECT_HERE_STRING, "REDIRECT_HERE_STRING"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum State {
    Start,
    Document,
    Statement,
    PipelineCommand,
    CommandName,
    CommandNameBytes,
    CommandBody,
    ArgumentBytes,
    RedirectBody,
    RedirectDescriptorBytes,
    RedirectOperator,
    RedirectTarget,
    RedirectTargetBytes,
    AfterRedirect,
    CommandRedirects,
    AfterCommand,
    DeclarationUtility,
    DeclarationUtilityBytes,
    DeclarationBody,
    DeclarationOptionBytes,
    AssignmentName,
    AssignmentNameBytes,
    AssignmentBody,
    AssignmentValueBytes,
    DeclarationAssignments,
    AfterDocument,
    Complete,
}

impl State {
    const ALL: [Self; 27] = [
        Self::Start,
        Self::Document,
        Self::Statement,
        Self::PipelineCommand,
        Self::CommandName,
        Self::CommandNameBytes,
        Self::CommandBody,
        Self::ArgumentBytes,
        Self::RedirectBody,
        Self::RedirectDescriptorBytes,
        Self::RedirectOperator,
        Self::RedirectTarget,
        Self::RedirectTargetBytes,
        Self::AfterRedirect,
        Self::CommandRedirects,
        Self::AfterCommand,
        Self::DeclarationUtility,
        Self::DeclarationUtilityBytes,
        Self::DeclarationBody,
        Self::DeclarationOptionBytes,
        Self::AssignmentName,
        Self::AssignmentNameBytes,
        Self::AssignmentBody,
        Self::AssignmentValueBytes,
        Self::DeclarationAssignments,
        Self::AfterDocument,
        Self::Complete,
    ];

    const fn id(self) -> u8 {
        self as u8
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Document => "document",
            Self::Statement => "statement",
            Self::PipelineCommand => "pipeline_command",
            Self::CommandName => "command_name",
            Self::CommandNameBytes => "command_name_bytes",
            Self::CommandBody => "command_body",
            Self::ArgumentBytes => "argument_bytes",
            Self::RedirectBody => "redirect_body",
            Self::RedirectDescriptorBytes => "redirect_descriptor_bytes",
            Self::RedirectOperator => "redirect_operator",
            Self::RedirectTarget => "redirect_target",
            Self::RedirectTargetBytes => "redirect_target_bytes",
            Self::AfterRedirect => "after_redirect",
            Self::CommandRedirects => "command_redirects",
            Self::AfterCommand => "after_command",
            Self::DeclarationUtility => "declaration_utility",
            Self::DeclarationUtilityBytes => "declaration_utility_bytes",
            Self::DeclarationBody => "declaration_body",
            Self::DeclarationOptionBytes => "declaration_option_bytes",
            Self::AssignmentName => "assignment_name",
            Self::AssignmentNameBytes => "assignment_name_bytes",
            Self::AssignmentBody => "assignment_body",
            Self::AssignmentValueBytes => "assignment_value_bytes",
            Self::DeclarationAssignments => "declaration_assignments",
            Self::AfterDocument => "after_document",
            Self::Complete => "complete",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SemanticActionV1;

impl SemanticActionV1 {
    pub fn encode(document: &SemanticDocumentV2) -> Result<Vec<u16>, ActionError> {
        document
            .validate()
            .map_err(|error| ActionError::Semantic(error.to_string()))?;
        let mut output = vec![token::BOS, token::DOC_START];
        for statement in &document.statements {
            match statement {
                StatementV2::Pipeline { stages, operators } => {
                    output.push(token::PIPELINE_START);
                    for (index, command) in stages.iter().enumerate() {
                        if index > 0 {
                            output.push(match operators[index - 1] {
                                PipelineOperatorV1::Stdout => token::PIPE_STDOUT,
                                PipelineOperatorV1::StdoutAndStderr => token::PIPE_STDOUT_STDERR,
                            });
                        }
                        encode_command(command, &mut output);
                    }
                    output.push(token::PIPELINE_END);
                }
                StatementV2::Declaration {
                    utility,
                    options,
                    assignments,
                } => {
                    output.push(token::DECLARATION_START);
                    encode_word(token::UTILITY_START, utility.source(), &mut output);
                    for option in options {
                        encode_word(token::OPTION_START, option.source(), &mut output);
                    }
                    for assignment in assignments {
                        output.push(token::ASSIGNMENT_START);
                        encode_word(token::ASSIGNMENT_NAME_START, &assignment.name, &mut output);
                        if let Some(value) = &assignment.value {
                            encode_word(token::ASSIGNMENT_VALUE_START, value.source(), &mut output);
                        }
                        output.push(token::ASSIGNMENT_END);
                    }
                    output.push(token::DECLARATION_END);
                }
            }
        }
        output.extend([token::DOC_END, token::EOS]);
        validate_tokens(&output)?;
        Ok(output)
    }

    pub fn decode(tokens: &[u16]) -> Result<SemanticDocumentV2, ActionError> {
        validate_tokens(tokens)?;
        let mut cursor = Cursor::new(tokens);
        cursor.expect(token::BOS)?;
        cursor.expect(token::DOC_START)?;
        let mut statements = Vec::new();
        while cursor.peek()? != token::DOC_END {
            statements.push(match cursor.peek()? {
                token::PIPELINE_START => decode_pipeline(&mut cursor)?,
                token::DECLARATION_START => decode_declaration(&mut cursor)?,
                found => return Err(cursor.unexpected(found)),
            });
        }
        cursor.expect(token::DOC_END)?;
        cursor.expect(token::EOS)?;
        if cursor.position != tokens.len() {
            return Err(ActionError::TrailingTokens {
                position: cursor.position,
            });
        }
        let document = SemanticDocumentV2 {
            version: SEMANTIC_SCHEMA_VERSION_V2,
            dialect: ShellDialect::Zsh,
            statements,
        };
        document
            .validate()
            .map_err(|error| ActionError::Semantic(error.to_string()))?;
        Ok(document)
    }

    pub fn grammar_manifest() -> GrammarManifest {
        GrammarManifest {
            schema_version: ACTION_SCHEMA_VERSION,
            semantic_schema_version: SEMANTIC_SCHEMA_VERSION_V2,
            vocab_size: ACTION_VOCAB_SIZE,
            byte_offset: BYTE_OFFSET,
            byte_count: 256,
            start_state: State::Start.id(),
            complete_state: State::Complete.id(),
            tokens: TOKEN_NAMES.iter().map(|(id, name)| NamedToken { id: *id, name }).collect(),
            states: State::ALL.iter().map(|state| manifest_state(*state)).collect(),
        }
    }
}

fn encode_command(command: &CommandV2, output: &mut Vec<u16>) {
    output.push(token::COMMAND_START);
    encode_word(token::COMMAND_NAME_START, command.name.source(), output);
    for argument in &command.arguments {
        encode_word(token::ARGUMENT_START, argument.source(), output);
    }
    for redirect in &command.redirects {
        output.push(token::REDIRECT_START);
        if let Some(descriptor) = &redirect.descriptor {
            encode_word(token::REDIRECT_DESCRIPTOR_START, descriptor, output);
        }
        output.push(redirect_token(redirect.operator));
        encode_word(token::REDIRECT_TARGET_START, redirect.target.source(), output);
        output.push(token::REDIRECT_END);
    }
    output.push(token::COMMAND_END);
}

fn encode_word(start: u16, value: &str, output: &mut Vec<u16>) {
    output.push(start);
    output.extend(value.as_bytes().iter().map(|byte| BYTE_OFFSET + u16::from(*byte)));
    output.push(token::WORD_END);
}

fn redirect_token(operator: RedirectOperatorV2) -> u16 {
    match operator {
        RedirectOperatorV2::Read => token::REDIRECT_READ,
        RedirectOperatorV2::Write => token::REDIRECT_WRITE,
        RedirectOperatorV2::Append => token::REDIRECT_APPEND,
        RedirectOperatorV2::Clobber => token::REDIRECT_CLOBBER,
        RedirectOperatorV2::ReadWrite => token::REDIRECT_READ_WRITE,
        RedirectOperatorV2::DuplicateInput => token::REDIRECT_DUPLICATE_INPUT,
        RedirectOperatorV2::DuplicateOutput => token::REDIRECT_DUPLICATE_OUTPUT,
        RedirectOperatorV2::HereString => token::REDIRECT_HERE_STRING,
    }
}

fn decode_pipeline(cursor: &mut Cursor<'_>) -> Result<StatementV2, ActionError> {
    cursor.expect(token::PIPELINE_START)?;
    let mut stages = vec![decode_command(cursor)?];
    let mut operators = Vec::new();
    while cursor.peek()? != token::PIPELINE_END {
        operators.push(match cursor.take()? {
            token::PIPE_STDOUT => PipelineOperatorV1::Stdout,
            token::PIPE_STDOUT_STDERR => PipelineOperatorV1::StdoutAndStderr,
            found => return Err(cursor.unexpected_at(found, cursor.position - 1)),
        });
        stages.push(decode_command(cursor)?);
    }
    cursor.expect(token::PIPELINE_END)?;
    Ok(StatementV2::Pipeline { stages, operators })
}

fn decode_command(cursor: &mut Cursor<'_>) -> Result<CommandV2, ActionError> {
    cursor.expect(token::COMMAND_START)?;
    let name = WordV1 {
        source: cursor.word(token::COMMAND_NAME_START)?,
    };
    let mut arguments = Vec::new();
    while cursor.peek()? == token::ARGUMENT_START {
        arguments.push(WordV1 {
            source: cursor.word(token::ARGUMENT_START)?,
        });
    }
    let mut redirects = Vec::new();
    while cursor.peek()? == token::REDIRECT_START {
        cursor.take()?;
        let descriptor = if cursor.peek()? == token::REDIRECT_DESCRIPTOR_START {
            Some(cursor.word(token::REDIRECT_DESCRIPTOR_START)?)
        } else {
            None
        };
        let operator = match cursor.take()? {
            token::REDIRECT_READ => RedirectOperatorV2::Read,
            token::REDIRECT_WRITE => RedirectOperatorV2::Write,
            token::REDIRECT_APPEND => RedirectOperatorV2::Append,
            token::REDIRECT_CLOBBER => RedirectOperatorV2::Clobber,
            token::REDIRECT_READ_WRITE => RedirectOperatorV2::ReadWrite,
            token::REDIRECT_DUPLICATE_INPUT => RedirectOperatorV2::DuplicateInput,
            token::REDIRECT_DUPLICATE_OUTPUT => RedirectOperatorV2::DuplicateOutput,
            token::REDIRECT_HERE_STRING => RedirectOperatorV2::HereString,
            found => return Err(cursor.unexpected_at(found, cursor.position - 1)),
        };
        let target = WordV1 {
            source: cursor.word(token::REDIRECT_TARGET_START)?,
        };
        cursor.expect(token::REDIRECT_END)?;
        redirects.push(RedirectV2 {
            descriptor,
            operator,
            target,
        });
    }
    cursor.expect(token::COMMAND_END)?;
    Ok(CommandV2 {
        name,
        arguments,
        redirects,
    })
}

fn decode_declaration(cursor: &mut Cursor<'_>) -> Result<StatementV2, ActionError> {
    cursor.expect(token::DECLARATION_START)?;
    let utility = WordV1 {
        source: cursor.word(token::UTILITY_START)?,
    };
    let mut options = Vec::new();
    while cursor.peek()? == token::OPTION_START {
        options.push(WordV1 {
            source: cursor.word(token::OPTION_START)?,
        });
    }
    let mut assignments = Vec::new();
    while cursor.peek()? == token::ASSIGNMENT_START {
        cursor.take()?;
        let name = cursor.word(token::ASSIGNMENT_NAME_START)?;
        let value = if cursor.peek()? == token::ASSIGNMENT_VALUE_START {
            Some(WordV1 {
                source: cursor.word(token::ASSIGNMENT_VALUE_START)?,
            })
        } else {
            None
        };
        cursor.expect(token::ASSIGNMENT_END)?;
        assignments.push(AssignmentV2 { name, value });
    }
    cursor.expect(token::DECLARATION_END)?;
    Ok(StatementV2::Declaration {
        utility,
        options,
        assignments,
    })
}

#[derive(Clone, Debug)]
struct Grammar {
    state: State,
    bytes: Vec<u8>,
}

impl Grammar {
    fn new() -> Self {
        Self {
            state: State::Start,
            bytes: Vec::new(),
        }
    }

    fn push(&mut self, value: u16, position: usize) -> Result<(), ActionError> {
        if is_byte_state(self.state) {
            if (BYTE_OFFSET..ACTION_VOCAB_SIZE).contains(&value) {
                self.bytes.push((value - BYTE_OFFSET) as u8);
                if valid_utf8_prefix(&self.bytes) {
                    return Ok(());
                }
                return Err(ActionError::InvalidUtf8 { position });
            }
            if value == token::WORD_END && !self.bytes.is_empty() && std::str::from_utf8(&self.bytes).is_ok() {
                self.bytes.clear();
                self.state = word_next(self.state);
                return Ok(());
            }
            return Err(ActionError::InvalidTransition {
                state: self.state.name(),
                token: value,
                position,
            });
        }
        let Some(next) = fixed_transition(self.state, value) else {
            return Err(ActionError::InvalidTransition {
                state: self.state.name(),
                token: value,
                position,
            });
        };
        self.state = next;
        if is_byte_state(next) {
            self.bytes.clear();
        }
        Ok(())
    }
}

fn validate_tokens(tokens: &[u16]) -> Result<(), ActionError> {
    let mut grammar = Grammar::new();
    for (position, value) in tokens.iter().copied().enumerate() {
        if value >= ACTION_VOCAB_SIZE {
            return Err(ActionError::OutOfVocabulary { token: value, position });
        }
        grammar.push(value, position)?;
    }
    if grammar.state == State::Complete {
        Ok(())
    } else if is_byte_state(grammar.state) && std::str::from_utf8(&grammar.bytes).is_err() {
        Err(ActionError::IncompleteUtf8 { position: tokens.len() })
    } else {
        Err(ActionError::Incomplete {
            state: grammar.state.name(),
            position: tokens.len(),
        })
    }
}

fn fixed_transition(state: State, value: u16) -> Option<State> {
    use State::*;
    Some(match (state, value) {
        (Start, token::BOS) => Document,
        (Document, token::DOC_START) => Statement,
        (Statement, token::PIPELINE_START) => PipelineCommand,
        (Statement, token::DECLARATION_START) => DeclarationUtility,
        (Statement, token::DOC_END) => AfterDocument,
        (PipelineCommand, token::COMMAND_START) => CommandName,
        (CommandName, token::COMMAND_NAME_START) => CommandNameBytes,
        (CommandBody, token::ARGUMENT_START) => ArgumentBytes,
        (CommandBody, token::REDIRECT_START) => RedirectBody,
        (CommandBody, token::COMMAND_END) => AfterCommand,
        (RedirectBody, token::REDIRECT_DESCRIPTOR_START) => RedirectDescriptorBytes,
        (RedirectBody, value) | (RedirectOperator, value) if is_redirect_token(value) => RedirectTarget,
        (RedirectTarget, token::REDIRECT_TARGET_START) => RedirectTargetBytes,
        (AfterRedirect, token::REDIRECT_END) => CommandRedirects,
        (CommandRedirects, token::REDIRECT_START) => RedirectBody,
        (CommandRedirects, token::COMMAND_END) => AfterCommand,
        (AfterCommand, token::PIPE_STDOUT) | (AfterCommand, token::PIPE_STDOUT_STDERR) => PipelineCommand,
        (AfterCommand, token::PIPELINE_END) => Statement,
        (DeclarationUtility, token::UTILITY_START) => DeclarationUtilityBytes,
        (DeclarationBody, token::OPTION_START) => DeclarationOptionBytes,
        (DeclarationBody, token::ASSIGNMENT_START) => AssignmentName,
        (AssignmentName, token::ASSIGNMENT_NAME_START) => AssignmentNameBytes,
        (AssignmentBody, token::ASSIGNMENT_VALUE_START) => AssignmentValueBytes,
        (AssignmentBody, token::ASSIGNMENT_END) => DeclarationAssignments,
        (DeclarationAssignments, token::ASSIGNMENT_START) => AssignmentName,
        (DeclarationAssignments, token::DECLARATION_END) => Statement,
        (AfterDocument, token::EOS) => Complete,
        _ => return None,
    })
}

fn is_redirect_token(value: u16) -> bool {
    (token::REDIRECT_READ..=token::REDIRECT_HERE_STRING).contains(&value)
}

fn is_byte_state(state: State) -> bool {
    matches!(
        state,
        State::CommandNameBytes
            | State::ArgumentBytes
            | State::RedirectDescriptorBytes
            | State::RedirectTargetBytes
            | State::DeclarationUtilityBytes
            | State::DeclarationOptionBytes
            | State::AssignmentNameBytes
            | State::AssignmentValueBytes
    )
}

fn word_next(state: State) -> State {
    match state {
        State::CommandNameBytes | State::ArgumentBytes => State::CommandBody,
        State::RedirectDescriptorBytes => State::RedirectOperator,
        State::RedirectTargetBytes => State::AfterRedirect,
        State::DeclarationUtilityBytes | State::DeclarationOptionBytes => State::DeclarationBody,
        State::AssignmentNameBytes | State::AssignmentValueBytes => State::AssignmentBody,
        _ => unreachable!("word_next called for non-byte state"),
    }
}

fn valid_utf8_prefix(bytes: &[u8]) -> bool {
    match std::str::from_utf8(bytes) {
        Ok(_) => true,
        Err(error) => error.error_len().is_none(),
    }
}

#[derive(Debug, Serialize)]
pub struct GrammarManifest {
    pub schema_version: u8,
    pub semantic_schema_version: u8,
    pub vocab_size: u16,
    pub byte_offset: u16,
    pub byte_count: u16,
    pub start_state: u8,
    pub complete_state: u8,
    pub tokens: Vec<NamedToken>,
    pub states: Vec<ManifestState>,
}

#[derive(Debug, Serialize)]
pub struct NamedToken {
    pub id: u16,
    pub name: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ManifestState {
    pub id: u8,
    pub name: &'static str,
    pub fixed: Vec<ManifestTransition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_payload: Option<BytePayload>,
}

#[derive(Debug, Serialize)]
pub struct ManifestTransition {
    pub token: u16,
    pub next_state: u8,
}

#[derive(Debug, Serialize)]
pub struct BytePayload {
    pub byte_offset: u16,
    pub byte_count: u16,
    pub end_token: u16,
    pub next_state: u8,
    pub require_nonempty: bool,
    pub require_valid_utf8: bool,
}

fn manifest_state(state: State) -> ManifestState {
    let fixed = (0..BYTE_OFFSET)
        .filter_map(|value| {
            fixed_transition(state, value).map(|next| ManifestTransition {
                token: value,
                next_state: next.id(),
            })
        })
        .collect();
    let byte_payload = is_byte_state(state).then(|| BytePayload {
        byte_offset: BYTE_OFFSET,
        byte_count: 256,
        end_token: token::WORD_END,
        next_state: word_next(state).id(),
        require_nonempty: true,
        require_valid_utf8: true,
    });
    ManifestState {
        id: state.id(),
        name: state.name(),
        fixed,
        byte_payload,
    }
}

struct Cursor<'a> {
    tokens: &'a [u16],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(tokens: &'a [u16]) -> Self {
        Self { tokens, position: 0 }
    }

    fn peek(&self) -> Result<u16, ActionError> {
        self.tokens
            .get(self.position)
            .copied()
            .ok_or(ActionError::UnexpectedEnd { position: self.position })
    }

    fn take(&mut self) -> Result<u16, ActionError> {
        let value = self.peek()?;
        self.position += 1;
        Ok(value)
    }

    fn expect(&mut self, expected: u16) -> Result<(), ActionError> {
        let found = self.take()?;
        if found == expected {
            Ok(())
        } else {
            Err(self.unexpected_at(found, self.position - 1))
        }
    }

    fn word(&mut self, start: u16) -> Result<String, ActionError> {
        self.expect(start)?;
        let begin = self.position;
        while self.peek()? != token::WORD_END {
            self.position += 1;
        }
        let bytes = self.tokens[begin..self.position]
            .iter()
            .map(|value| (value - BYTE_OFFSET) as u8)
            .collect::<Vec<_>>();
        self.expect(token::WORD_END)?;
        String::from_utf8(bytes).map_err(|_| ActionError::InvalidUtf8 { position: begin })
    }

    fn unexpected(&self, token: u16) -> ActionError {
        self.unexpected_at(token, self.position)
    }

    fn unexpected_at(&self, token: u16, position: usize) -> ActionError {
        ActionError::Decode { token, position }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActionError {
    Semantic(String),
    OutOfVocabulary {
        token: u16,
        position: usize,
    },
    InvalidTransition {
        state: &'static str,
        token: u16,
        position: usize,
    },
    InvalidUtf8 {
        position: usize,
    },
    IncompleteUtf8 {
        position: usize,
    },
    Incomplete {
        state: &'static str,
        position: usize,
    },
    UnexpectedEnd {
        position: usize,
    },
    Decode {
        token: u16,
        position: usize,
    },
    TrailingTokens {
        position: usize,
    },
}

impl fmt::Display for ActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Semantic(message) => write!(formatter, "semantic validation failed: {message}"),
            Self::OutOfVocabulary { token, position } => {
                write!(formatter, "token {token} at {position} is outside the action vocabulary")
            }
            Self::InvalidTransition { state, token, position } => {
                write!(formatter, "token {token} is invalid in grammar state {state} at {position}")
            }
            Self::InvalidUtf8 { position } => write!(formatter, "invalid UTF-8 byte sequence at {position}"),
            Self::IncompleteUtf8 { position } => write!(formatter, "incomplete UTF-8 byte sequence at {position}"),
            Self::Incomplete { state, position } => {
                write!(formatter, "incomplete action sequence in state {state} at {position}")
            }
            Self::UnexpectedEnd { position } => write!(formatter, "unexpected end of action sequence at {position}"),
            Self::Decode { token, position } => write!(formatter, "unexpected token {token} while decoding at {position}"),
            Self::TrailingTokens { position } => write!(formatter, "trailing action tokens at {position}"),
        }
    }
}

impl std::error::Error for ActionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SyntaxDocumentV1;

    fn semantic(source: &str) -> SemanticDocumentV2 {
        let syntax = SyntaxDocumentV1::parse(source).unwrap_or_else(|error| panic!("syntax for {source:?}: {error}"));
        SemanticDocumentV2::lower(&syntax).unwrap_or_else(|error| panic!("semantic for {source:?}: {error}"))
    }

    fn round_trip(source: &str) {
        let expected = semantic(source);
        let tokens = SemanticActionV1::encode(&expected).expect("encode");
        let actual = SemanticActionV1::decode(&tokens).expect("decode");
        assert_eq!(actual, expected);
        assert_eq!(actual.render(), expected.render());
    }

    #[test]
    fn round_trips_all_supported_structures() {
        round_trip("printf '%s\\n' café |& grep fé >out\nexport -n FOO=bar EMPTY");
        for source in [
            "cat <input",
            "cat >out",
            "cat 2>>err",
            "cat >|force",
            "cat <&0",
            "cat >&1",
            "cat <<<word",
        ] {
            round_trip(source);
        }
        round_trip("one a | two b |& three c");
    }

    #[test]
    fn manifest_covers_compact_vocabulary_and_all_states() {
        let manifest = SemanticActionV1::grammar_manifest();
        assert_eq!(manifest.vocab_size, 320);
        assert_eq!(manifest.byte_offset + manifest.byte_count, manifest.vocab_size);
        assert_eq!(manifest.states.len(), 27);
        assert!(manifest.states.iter().any(|state| state.byte_payload.is_some()));
        let redirect_body = manifest
            .states
            .iter()
            .find(|state| state.name == "redirect_body")
            .expect("redirect body");
        for redirect in token::REDIRECT_READ..=token::REDIRECT_HERE_STRING {
            assert!(redirect_body.fixed.iter().any(|edge| edge.token == redirect));
        }
    }

    #[test]
    fn malformed_sequences_fail_closed_by_category() {
        let valid = SemanticActionV1::encode(&semantic("echo café")).expect("encode");
        let mut invalid_transition = valid.clone();
        invalid_transition[2] = token::COMMAND_START;
        assert!(matches!(
            SemanticActionV1::decode(&invalid_transition),
            Err(ActionError::InvalidTransition { .. })
        ));

        let mut out_of_vocabulary = valid.clone();
        out_of_vocabulary[3] = ACTION_VOCAB_SIZE;
        assert!(matches!(
            SemanticActionV1::decode(&out_of_vocabulary),
            Err(ActionError::OutOfVocabulary { .. })
        ));

        let incomplete = &valid[..valid.len() - 1];
        assert!(matches!(
            SemanticActionV1::decode(incomplete),
            Err(ActionError::Incomplete { .. })
        ));

        let invalid_utf8 = vec![
            token::BOS,
            token::DOC_START,
            token::PIPELINE_START,
            token::COMMAND_START,
            token::COMMAND_NAME_START,
            BYTE_OFFSET + 0xff,
        ];
        assert!(matches!(
            SemanticActionV1::decode(&invalid_utf8),
            Err(ActionError::InvalidUtf8 { .. })
        ));

        let incomplete_utf8 = vec![
            token::BOS,
            token::DOC_START,
            token::PIPELINE_START,
            token::COMMAND_START,
            token::COMMAND_NAME_START,
            BYTE_OFFSET + 0xc3,
        ];
        assert!(matches!(
            SemanticActionV1::decode(&incomplete_utf8),
            Err(ActionError::IncompleteUtf8 { .. })
        ));

        let argument_after_redirect = SemanticActionV1::encode(&semantic("cat <input")).expect("encode redirect");
        let redirect_end = argument_after_redirect
            .iter()
            .position(|value| *value == token::REDIRECT_END)
            .expect("redirect end");
        let mut reordered = argument_after_redirect;
        reordered.splice(
            redirect_end + 1..redirect_end + 1,
            [token::ARGUMENT_START, BYTE_OFFSET + u16::from(b'x'), token::WORD_END],
        );
        assert!(matches!(
            SemanticActionV1::decode(&reordered),
            Err(ActionError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn arbitrary_malformed_tokens_never_escape_validation() {
        let mut value = 0x5eed_u32;
        for length in 0..96 {
            let tokens = (0..length)
                .map(|_| {
                    value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (value % 384) as u16
                })
                .collect::<Vec<_>>();
            assert!(SemanticActionV1::decode(&tokens).is_err());
        }
    }
}
