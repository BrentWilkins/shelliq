//! Compact, project-owned semantic shell representation.
//!
//! The lossless CST remains the validation boundary. This module lowers the
//! first useful subset into a small vocabulary intended to grow independently
//! of tree-sitter's grammar shape.

use std::fmt;

use serde::{Deserialize, Serialize};
use tree_sitter::Node;

use super::{ShellDialect, SyntaxDocumentV1, SyntaxError, parse_zsh};

/// Serialized semantic-AST contract currently emitted and accepted by shelliq.
pub const SEMANTIC_SCHEMA_VERSION: u8 = 1;

/// A versioned semantic shell document.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticDocumentV1 {
    /// Schema version. Must equal [`SEMANTIC_SCHEMA_VERSION`].
    #[serde(rename = "v")]
    version: u8,
    /// Shell dialect used to interpret words and operators.
    #[serde(rename = "d")]
    dialect: ShellDialect,
    /// Sequential statements.
    #[serde(rename = "s")]
    statements: Vec<StatementV1>,
}

/// Semantic-AST v2 adds declarations and here-string redirects while leaving
/// the v1 wire contract available for existing consumers.
pub const SEMANTIC_SCHEMA_VERSION_V2: u8 = 2;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticDocumentV2 {
    #[serde(rename = "v")]
    version: u8,
    #[serde(rename = "d")]
    dialect: ShellDialect,
    #[serde(rename = "s")]
    statements: Vec<StatementV2>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "t", deny_unknown_fields)]
pub enum StatementV2 {
    #[serde(rename = "p")]
    Pipeline {
        #[serde(rename = "c")]
        stages: Vec<CommandV2>,
        #[serde(rename = "o", default, skip_serializing_if = "Vec::is_empty")]
        operators: Vec<PipelineOperatorV1>,
    },
    #[serde(rename = "d")]
    Declaration {
        #[serde(rename = "u")]
        utility: WordV1,
        #[serde(rename = "o", default, skip_serializing_if = "Vec::is_empty")]
        options: Vec<WordV1>,
        #[serde(rename = "a")]
        assignments: Vec<AssignmentV2>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentV2 {
    #[serde(rename = "n")]
    name: String,
    #[serde(rename = "v", default, skip_serializing_if = "Option::is_none")]
    value: Option<WordV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommandV2 {
    #[serde(rename = "n")]
    name: WordV1,
    #[serde(rename = "a", default, skip_serializing_if = "Vec::is_empty")]
    arguments: Vec<WordV1>,
    #[serde(rename = "r", default, skip_serializing_if = "Vec::is_empty")]
    redirects: Vec<RedirectV2>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedirectV2 {
    #[serde(rename = "f", default, skip_serializing_if = "Option::is_none")]
    descriptor: Option<String>,
    #[serde(rename = "o")]
    operator: RedirectOperatorV2,
    #[serde(rename = "t")]
    target: WordV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RedirectOperatorV2 {
    #[serde(rename = "<")]
    Read,
    #[serde(rename = ">")]
    Write,
    #[serde(rename = ">>")]
    Append,
    #[serde(rename = ">|")]
    Clobber,
    #[serde(rename = "<>")]
    ReadWrite,
    #[serde(rename = "<&")]
    DuplicateInput,
    #[serde(rename = ">&")]
    DuplicateOutput,
    #[serde(rename = "<<<")]
    HereString,
}

impl SemanticDocumentV2 {
    pub fn lower(document: &SyntaxDocumentV1) -> Result<Self, SemanticError> {
        document.validate()?;
        Self::lower_source(&document.render())
    }

    pub const fn version(&self) -> u8 {
        self.version
    }

    pub fn statements(&self) -> &[StatementV2] {
        &self.statements
    }

    /// First executable name, for retrieval routing before final generation.
    pub fn first_command_name(&self) -> Option<&str> {
        match self.statements.first()? {
            StatementV2::Pipeline { stages, .. } => Some(stages.first()?.name().source()),
            StatementV2::Declaration { utility, .. } => Some(utility.source()),
        }
    }

    pub fn render(&self) -> String {
        self.statements.iter().map(StatementV2::render).collect::<Vec<_>>().join("\n")
    }

    pub fn validate(&self) -> Result<(), SemanticError> {
        if self.version != SEMANTIC_SCHEMA_VERSION_V2 {
            return Err(SemanticError::UnsupportedVersion(self.version));
        }
        let lowered = Self::lower_source(&self.render())?;
        if lowered == *self {
            Ok(())
        } else {
            Err(SemanticError::StructureMismatch)
        }
    }

    fn lower_source(source: &str) -> Result<Self, SemanticError> {
        let tree = parse_zsh(source)?;
        let root = tree.root_node();
        let mut cursor = root.walk();
        let statements = root
            .named_children(&mut cursor)
            .map(|child| lower_statement_v2(child, source))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            version: SEMANTIC_SCHEMA_VERSION_V2,
            dialect: ShellDialect::Zsh,
            statements,
        })
    }
}

impl StatementV2 {
    fn render(&self) -> String {
        match self {
            Self::Pipeline { stages, operators } => {
                let mut rendered = String::new();
                for (index, stage) in stages.iter().enumerate() {
                    if let Some(operator) = index.checked_sub(1).and_then(|i| operators.get(i)) {
                        rendered.push(' ');
                        rendered.push_str(operator.source());
                        rendered.push(' ');
                    }
                    stage.render_into(&mut rendered);
                }
                rendered
            }
            Self::Declaration {
                utility,
                options,
                assignments,
            } => {
                let mut rendered = utility.source().to_owned();
                for option in options {
                    rendered.push(' ');
                    rendered.push_str(option.source());
                }
                for assignment in assignments {
                    rendered.push(' ');
                    rendered.push_str(&assignment.name);
                    if let Some(value) = &assignment.value {
                        rendered.push('=');
                        rendered.push_str(value.source());
                    }
                }
                rendered
            }
        }
    }
}

impl AssignmentV2 {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn value(&self) -> Option<&WordV1> {
        self.value.as_ref()
    }
}

impl CommandV2 {
    pub const fn name(&self) -> &WordV1 {
        &self.name
    }

    pub fn arguments(&self) -> &[WordV1] {
        &self.arguments
    }

    pub fn redirects(&self) -> &[RedirectV2] {
        &self.redirects
    }

    fn render_into(&self, rendered: &mut String) {
        rendered.push_str(self.name.source());
        for argument in &self.arguments {
            rendered.push(' ');
            rendered.push_str(argument.source());
        }
        for redirect in &self.redirects {
            rendered.push(' ');
            if let Some(descriptor) = &redirect.descriptor {
                rendered.push_str(descriptor);
            }
            rendered.push_str(redirect.operator.source());
            rendered.push_str(redirect.target.source());
        }
    }
}

impl RedirectV2 {
    pub fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }

    pub const fn operator(&self) -> RedirectOperatorV2 {
        self.operator
    }

    pub const fn target(&self) -> &WordV1 {
        &self.target
    }
}

impl RedirectOperatorV2 {
    const fn source(self) -> &'static str {
        match self {
            Self::Read => "<",
            Self::Write => ">",
            Self::Append => ">>",
            Self::Clobber => ">|",
            Self::ReadWrite => "<>",
            Self::DuplicateInput => "<&",
            Self::DuplicateOutput => ">&",
            Self::HereString => "<<<",
        }
    }
}

fn lower_statement_v2(node: Node<'_>, source: &str) -> Result<StatementV2, SemanticError> {
    match node.kind() {
        "command" => Ok(StatementV2::Pipeline {
            stages: vec![lower_stage_v2(node, source)?],
            operators: Vec::new(),
        }),
        "redirected_statement" => lower_redirected_statement_v2(node, source),
        "pipeline" => lower_pipeline_v2(node, source),
        "declaration_command" => lower_declaration_v2(node, source),
        _ => Err(unsupported(node)),
    }
}

fn lower_redirected_statement_v2(node: Node<'_>, source: &str) -> Result<StatementV2, SemanticError> {
    let body = node.child_by_field_name("body").ok_or_else(|| unsupported(node))?;
    let mut statement = lower_statement_v2(body, source)?;
    let StatementV2::Pipeline { stages, .. } = &mut statement else {
        return Err(unsupported(body));
    };
    let last_stage = stages.last_mut().ok_or_else(|| unsupported(node))?;
    let mut cursor = node.walk();
    for redirect in node.children_by_field_name("redirect", &mut cursor) {
        last_stage.redirects.push(lower_redirect_v2(redirect, source)?);
    }
    Ok(statement)
}

fn lower_pipeline_v2(node: Node<'_>, source: &str) -> Result<StatementV2, SemanticError> {
    let mut stages = Vec::new();
    let mut operators = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "command" | "redirected_statement" => stages.push(lower_stage_v2(child, source)?),
            "|" => operators.push(PipelineOperatorV1::Stdout),
            "|&" => operators.push(PipelineOperatorV1::StdoutAndStderr),
            _ if !child.is_named() => return Err(unsupported(child)),
            _ => return Err(unsupported(child)),
        }
    }
    if stages.is_empty() || operators.len() + 1 != stages.len() {
        return Err(unsupported(node));
    }
    Ok(StatementV2::Pipeline { stages, operators })
}

fn lower_stage_v2(node: Node<'_>, source: &str) -> Result<CommandV2, SemanticError> {
    if node.kind() == "command" {
        return lower_command_v2(node, source);
    }
    let body = node.child_by_field_name("body").ok_or_else(|| unsupported(node))?;
    if body.kind() == "pipeline" {
        return Err(unsupported(body));
    }
    let mut command = lower_stage_v2(body, source)?;
    let mut cursor = node.walk();
    for redirect in node.children_by_field_name("redirect", &mut cursor) {
        command.redirects.push(lower_redirect_v2(redirect, source)?);
    }
    Ok(command)
}

fn lower_command_v2(node: Node<'_>, source: &str) -> Result<CommandV2, SemanticError> {
    let name = node.child_by_field_name("name").ok_or_else(|| unsupported(node))?;
    let mut cursor = node.walk();
    let arguments = node
        .children_by_field_name("argument", &mut cursor)
        .map(|argument| word(argument, source))
        .collect::<Vec<_>>();
    if node.named_child_count() != 1 + arguments.len() {
        return Err(unsupported(node));
    }
    Ok(CommandV2 {
        name: word(name, source),
        arguments,
        redirects: Vec::new(),
    })
}

fn lower_redirect_v2(node: Node<'_>, source: &str) -> Result<RedirectV2, SemanticError> {
    if node.kind() == "herestring_redirect" {
        let operator = node.child(0).ok_or_else(|| unsupported(node))?;
        let target = node.named_child(0).ok_or_else(|| unsupported(node))?;
        if operator.kind() != "<<<" {
            return Err(unsupported(node));
        }
        return Ok(RedirectV2 {
            descriptor: None,
            operator: RedirectOperatorV2::HereString,
            target: word(target, source),
        });
    }
    if node.kind() != "file_redirect" {
        return Err(unsupported(node));
    }
    let descriptor = node
        .child_by_field_name("descriptor")
        .map(|child| text(child, source).to_owned());
    let target = node.child_by_field_name("destination").ok_or_else(|| unsupported(node))?;
    let mut operator = None;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        operator = match child.kind() {
            "<" => Some(RedirectOperatorV2::Read),
            ">" => Some(RedirectOperatorV2::Write),
            ">>" => Some(RedirectOperatorV2::Append),
            ">|" => Some(RedirectOperatorV2::Clobber),
            "<>" => Some(RedirectOperatorV2::ReadWrite),
            "<&" => Some(RedirectOperatorV2::DuplicateInput),
            ">&" => Some(RedirectOperatorV2::DuplicateOutput),
            _ => operator,
        };
    }
    Ok(RedirectV2 {
        descriptor,
        operator: operator.ok_or_else(|| unsupported(node))?,
        target: word(target, source),
    })
}

fn lower_declaration_v2(node: Node<'_>, source: &str) -> Result<StatementV2, SemanticError> {
    let utility_node = node.child(0).ok_or_else(|| unsupported(node))?;
    let mut argument_cursor = node.walk();
    let options = node
        .children_by_field_name("argument", &mut argument_cursor)
        .map(|argument| word(argument, source))
        .collect::<Vec<_>>();
    let mut assignments = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() != "variable_assignment" {
            continue;
        }
        let name = child.child_by_field_name("name").ok_or_else(|| unsupported(child))?;
        let value = child.child_by_field_name("value").map(|value| word(value, source));
        assignments.push(AssignmentV2 {
            name: text(name, source).to_owned(),
            value,
        });
    }
    if node.named_child_count() != options.len() + assignments.len() || assignments.is_empty() {
        return Err(unsupported(node));
    }
    Ok(StatementV2::Declaration {
        utility: word(utility_node, source),
        options,
        assignments,
    })
}

/// A pipeline. A simple command is represented by one stage.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatementV1 {
    /// Commands in execution order.
    #[serde(rename = "c")]
    stages: Vec<CommandV1>,
    /// Operators between adjacent stages.
    #[serde(rename = "o", default, skip_serializing_if = "Vec::is_empty")]
    operators: Vec<PipelineOperatorV1>,
}

/// One simple command and any redirects attached to it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommandV1 {
    /// Command name.
    #[serde(rename = "n")]
    name: WordV1,
    /// Ordered command arguments.
    #[serde(rename = "a", default, skip_serializing_if = "Vec::is_empty")]
    arguments: Vec<WordV1>,
    /// Ordered file redirects.
    #[serde(rename = "r", default, skip_serializing_if = "Vec::is_empty")]
    redirects: Vec<RedirectV1>,
}

/// One shell word, retained as a single lexical unit in the initial lowerer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WordV1 {
    /// Zsh source spelling for this one word.
    #[serde(rename = "s")]
    source: String,
}

/// A file redirect attached to a command.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedirectV1 {
    /// Optional explicit file descriptor, such as `2` in `2>>errors`.
    #[serde(rename = "f", default, skip_serializing_if = "Option::is_none")]
    descriptor: Option<String>,
    /// Redirect operator.
    #[serde(rename = "o")]
    operator: RedirectOperatorV1,
    /// Redirect destination.
    #[serde(rename = "t")]
    target: WordV1,
}

/// Pipeline connection semantics.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PipelineOperatorV1 {
    /// Pipe standard output.
    #[serde(rename = "|")]
    Stdout,
    /// Pipe standard output and standard error (Zsh `|&`).
    #[serde(rename = "|&")]
    StdoutAndStderr,
}

/// Supported file redirect operators.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RedirectOperatorV1 {
    #[serde(rename = "<")]
    Read,
    #[serde(rename = ">")]
    Write,
    #[serde(rename = ">>")]
    Append,
    #[serde(rename = ">|")]
    Clobber,
    #[serde(rename = "<>")]
    ReadWrite,
    #[serde(rename = "<&")]
    DuplicateInput,
    #[serde(rename = ">&")]
    DuplicateOutput,
}

/// Failure to lower or validate the compact semantic representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticError {
    /// The lossless Zsh syntax boundary rejected the source document.
    Syntax(SyntaxError),
    /// The CST contains a construct not covered by this lowering slice.
    Unsupported { byte: usize, kind: String },
    /// The serialized schema version is unknown.
    UnsupportedVersion(u8),
    /// Rendering and lowering again produced different semantics.
    StructureMismatch,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(error) => error.fmt(formatter),
            Self::Unsupported { byte, kind } => {
                write!(formatter, "unsupported semantic Zsh construct near byte {byte} ({kind})")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported semantic shell schema version {version}")
            }
            Self::StructureMismatch => {
                formatter.write_str("serialized semantic shell structure does not match rendered Zsh text")
            }
        }
    }
}

impl std::error::Error for SemanticError {}

impl From<SyntaxError> for SemanticError {
    fn from(value: SyntaxError) -> Self {
        Self::Syntax(value)
    }
}

impl SemanticDocumentV1 {
    /// Lower a validated lossless CST into the initial compact semantic subset.
    pub fn lower(document: &SyntaxDocumentV1) -> Result<Self, SemanticError> {
        document.validate()?;
        Self::lower_source(&document.render())
    }

    /// Return this document's schema version.
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// Return the semantic statements in execution order.
    pub fn statements(&self) -> &[StatementV1] {
        &self.statements
    }

    /// Render normalized Zsh source from the semantic representation.
    pub fn render(&self) -> String {
        self.statements.iter().map(StatementV1::render).collect::<Vec<_>>().join("\n")
    }

    /// Validate schema and semantic structure through render-and-reparse.
    pub fn validate(&self) -> Result<(), SemanticError> {
        if self.version != SEMANTIC_SCHEMA_VERSION {
            return Err(SemanticError::UnsupportedVersion(self.version));
        }
        let lowered = Self::lower_source(&self.render())?;
        if lowered != *self {
            return Err(SemanticError::StructureMismatch);
        }
        Ok(())
    }

    fn lower_source(source: &str) -> Result<Self, SemanticError> {
        let tree = parse_zsh(source)?;
        let root = tree.root_node();
        let mut statements = Vec::new();
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            statements.push(lower_statement(child, source)?);
        }
        Ok(Self {
            version: SEMANTIC_SCHEMA_VERSION,
            dialect: ShellDialect::Zsh,
            statements,
        })
    }
}

impl StatementV1 {
    /// Return pipeline stages in execution order.
    pub fn stages(&self) -> &[CommandV1] {
        &self.stages
    }

    /// Return the operators between pipeline stages.
    pub fn operators(&self) -> &[PipelineOperatorV1] {
        &self.operators
    }

    fn render(&self) -> String {
        let mut rendered = String::new();
        for (index, stage) in self.stages.iter().enumerate() {
            if let Some(operator) = index.checked_sub(1).and_then(|i| self.operators.get(i)) {
                rendered.push(' ');
                rendered.push_str(operator.source());
                rendered.push(' ');
            }
            stage.render_into(&mut rendered);
        }
        rendered
    }
}

impl CommandV1 {
    pub const fn name(&self) -> &WordV1 {
        &self.name
    }

    pub fn arguments(&self) -> &[WordV1] {
        &self.arguments
    }

    pub fn redirects(&self) -> &[RedirectV1] {
        &self.redirects
    }

    fn render_into(&self, rendered: &mut String) {
        rendered.push_str(self.name.source());
        for argument in &self.arguments {
            rendered.push(' ');
            rendered.push_str(argument.source());
        }
        for redirect in &self.redirects {
            rendered.push(' ');
            if let Some(descriptor) = &redirect.descriptor {
                rendered.push_str(descriptor);
            }
            rendered.push_str(redirect.operator.source());
            rendered.push_str(redirect.target.source());
        }
    }
}

impl WordV1 {
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl RedirectV1 {
    pub fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }

    pub const fn operator(&self) -> RedirectOperatorV1 {
        self.operator
    }

    pub const fn target(&self) -> &WordV1 {
        &self.target
    }
}

impl PipelineOperatorV1 {
    const fn source(self) -> &'static str {
        match self {
            Self::Stdout => "|",
            Self::StdoutAndStderr => "|&",
        }
    }
}

impl RedirectOperatorV1 {
    const fn source(self) -> &'static str {
        match self {
            Self::Read => "<",
            Self::Write => ">",
            Self::Append => ">>",
            Self::Clobber => ">|",
            Self::ReadWrite => "<>",
            Self::DuplicateInput => "<&",
            Self::DuplicateOutput => ">&",
        }
    }
}

fn lower_statement(node: Node<'_>, source: &str) -> Result<StatementV1, SemanticError> {
    match node.kind() {
        "command" => Ok(StatementV1 {
            stages: vec![lower_stage(node, source)?],
            operators: Vec::new(),
        }),
        "redirected_statement" => lower_redirected_statement(node, source),
        "pipeline" => lower_pipeline(node, source),
        _ => Err(unsupported(node)),
    }
}

fn lower_redirected_statement(node: Node<'_>, source: &str) -> Result<StatementV1, SemanticError> {
    let body = node.child_by_field_name("body").ok_or_else(|| unsupported(node))?;
    let mut statement = lower_statement(body, source)?;
    let last_stage = statement.stages.last_mut().ok_or_else(|| unsupported(body))?;
    let mut cursor = node.walk();
    for redirect in node.children_by_field_name("redirect", &mut cursor) {
        last_stage.redirects.push(lower_redirect(redirect, source)?);
    }
    Ok(statement)
}

fn lower_pipeline(node: Node<'_>, source: &str) -> Result<StatementV1, SemanticError> {
    let mut stages = Vec::new();
    let mut operators = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "command" | "redirected_statement" => stages.push(lower_stage(child, source)?),
            "|" => operators.push(PipelineOperatorV1::Stdout),
            "|&" => operators.push(PipelineOperatorV1::StdoutAndStderr),
            _ if !child.is_named() => return Err(unsupported(child)),
            _ => return Err(unsupported(child)),
        }
    }
    if stages.is_empty() || operators.len() + 1 != stages.len() {
        return Err(unsupported(node));
    }
    Ok(StatementV1 { stages, operators })
}

fn lower_stage(node: Node<'_>, source: &str) -> Result<CommandV1, SemanticError> {
    if node.kind() == "command" {
        return lower_command(node, source);
    }

    let body = node.child_by_field_name("body").ok_or_else(|| unsupported(node))?;
    if body.kind() == "pipeline" {
        return Err(unsupported(body));
    }
    let mut command = lower_stage(body, source)?;
    let mut cursor = node.walk();
    for redirect in node.children_by_field_name("redirect", &mut cursor) {
        command.redirects.push(lower_redirect(redirect, source)?);
    }
    Ok(command)
}

fn lower_command(node: Node<'_>, source: &str) -> Result<CommandV1, SemanticError> {
    let name = node.child_by_field_name("name").ok_or_else(|| unsupported(node))?;
    let mut arguments = Vec::new();
    let mut cursor = node.walk();
    for argument in node.children_by_field_name("argument", &mut cursor) {
        arguments.push(word(argument, source));
    }

    let allowed_named = 1 + arguments.len();
    if node.named_child_count() != allowed_named {
        return Err(unsupported(node));
    }
    Ok(CommandV1 {
        name: word(name, source),
        arguments,
        redirects: Vec::new(),
    })
}

fn lower_redirect(node: Node<'_>, source: &str) -> Result<RedirectV1, SemanticError> {
    if node.kind() != "file_redirect" {
        return Err(unsupported(node));
    }
    let descriptor = node
        .child_by_field_name("descriptor")
        .map(|child| text(child, source).to_owned());
    let target = node.child_by_field_name("destination").ok_or_else(|| unsupported(node))?;
    let mut operator = None;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.is_named() {
            continue;
        }
        operator = Some(match child.kind() {
            "<" => RedirectOperatorV1::Read,
            ">" => RedirectOperatorV1::Write,
            ">>" => RedirectOperatorV1::Append,
            ">|" => RedirectOperatorV1::Clobber,
            "<>" => RedirectOperatorV1::ReadWrite,
            "<&" => RedirectOperatorV1::DuplicateInput,
            ">&" => RedirectOperatorV1::DuplicateOutput,
            _ => return Err(unsupported(child)),
        });
    }
    Ok(RedirectV1 {
        descriptor,
        operator: operator.ok_or_else(|| unsupported(node))?,
        target: word(target, source),
    })
}

fn word(node: Node<'_>, source: &str) -> WordV1 {
    WordV1 {
        source: text(node, source).to_owned(),
    }
}

fn text<'a>(node: Node<'_>, source: &'a str) -> &'a str {
    &source[node.byte_range()]
}

fn unsupported(node: Node<'_>) -> SemanticError {
    SemanticError::Unsupported {
        byte: node.start_byte(),
        kind: node.kind().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower(source: &str) -> SemanticDocumentV1 {
        let syntax = SyntaxDocumentV1::parse(source).unwrap();
        SemanticDocumentV1::lower(&syntax).unwrap()
    }

    #[test]
    fn lowers_command_arguments_and_redirects() {
        let document = lower("print -r -- \"$name\" 2>>errors.log");
        let command = &document.statements()[0].stages()[0];
        assert_eq!(command.name().source(), "print");
        assert_eq!(
            command.arguments().iter().map(WordV1::source).collect::<Vec<_>>(),
            ["-r", "--", "\"$name\""]
        );
        assert_eq!(command.redirects()[0].descriptor(), Some("2"));
        assert_eq!(command.redirects()[0].operator(), RedirectOperatorV1::Append);
        assert_eq!(command.redirects()[0].target().source(), "errors.log");
        assert_eq!(document.render(), "print -r -- \"$name\" 2>>errors.log");
        document.validate().unwrap();
    }

    #[test]
    fn lowers_pipeline_operators() {
        let document = lower("find . -type f |& sort -u >files.txt");
        let statement = &document.statements()[0];
        assert_eq!(statement.stages().len(), 2);
        assert_eq!(statement.operators(), [PipelineOperatorV1::StdoutAndStderr]);
        assert_eq!(statement.stages()[1].redirects().len(), 1);
        assert_eq!(document.render(), "find . -type f |& sort -u >files.txt");
        document.validate().unwrap();
    }

    #[test]
    fn lowers_real_awk_word_without_flattening_quoting() {
        let source = "awk 'NR>=70 && NR<=110 {print NR \":X:\" $0}' training/SHELL_AST.md";
        let document = lower(source);
        assert_eq!(
            document.statements()[0].stages()[0].arguments()[0].source(),
            "'NR>=70 && NR<=110 {print NR \":X:\" $0}'"
        );
        assert_eq!(document.render(), source);
        document.validate().unwrap();
    }

    #[test]
    fn normalizes_sequential_statement_separators() {
        let document = lower("echo one; print two");
        assert_eq!(document.statements().len(), 2);
        assert_eq!(document.render(), "echo one\nprint two");
        document.validate().unwrap();
    }

    #[test]
    fn serde_round_trip_preserves_semantic_contract() {
        let document = lower("print -r -- **/*.rs(N.) | sort -u");
        let json = serde_json::to_string(&document).unwrap();
        let decoded: SemanticDocumentV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, document);
        decoded.validate().unwrap();
    }

    #[test]
    fn reports_the_first_executable_name_structurally() {
        let syntax = SyntaxDocumentV1::parse("find . -type f | sort -u").unwrap();
        let document = SemanticDocumentV2::lower(&syntax).unwrap();
        assert_eq!(document.first_command_name(), Some("find"));
    }

    #[test]
    fn rejects_control_flow_outside_initial_slice() {
        let syntax = SyntaxDocumentV1::parse("if true; then print ok; fi").unwrap();
        let error = SemanticDocumentV1::lower(&syntax).unwrap_err();
        assert!(matches!(error, SemanticError::Unsupported { .. }));
    }

    #[test]
    fn validation_rejects_word_that_changes_structure() {
        let mut document = lower("print ok");
        document.statements[0].stages[0].arguments[0].source = "ok; false".to_owned();
        assert_eq!(document.validate(), Err(SemanticError::StructureMismatch));
    }

    #[test]
    fn validation_rejects_unknown_schema() {
        let mut document = lower("print ok");
        document.version = 2;
        assert_eq!(document.validate(), Err(SemanticError::UnsupportedVersion(2)));
    }
}
