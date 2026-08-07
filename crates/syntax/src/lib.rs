//! Zsh-first, lossless shell syntax representation.
//!
//! This crate deliberately calls the representation a concrete syntax tree
//! (CST), not a semantic command AST. It preserves every byte and every Zsh
//! grammar node, so it can be the checked boundary around a later compact AST
//! without pretending that `command + flags + operands` is the whole shell
//! language.

use std::fmt;

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser, Tree};

pub mod semantic;

/// Serialized syntax contract currently emitted and accepted by shelliq.
pub const SYNTAX_SCHEMA_VERSION: u8 = 1;

/// Shell grammar used to interpret a syntax document.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ShellDialect {
    /// Zsh syntax as parsed by the pinned `tree-sitter-zsh` grammar.
    Zsh,
}

/// A versioned, lossless Zsh concrete syntax tree.
///
/// The compact serialized field names are intentional: this representation
/// may cross the model boundary, where repeated JSON keys consume context.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntaxDocumentV1 {
    /// Schema version. Must equal [`SYNTAX_SCHEMA_VERSION`].
    #[serde(rename = "v")]
    version: u8,
    /// Grammar dialect.
    #[serde(rename = "d")]
    dialect: ShellDialect,
    /// Root grammar node.
    #[serde(rename = "r")]
    root: SyntaxNodeV1,
}

/// One grammar node in a [`SyntaxDocumentV1`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntaxNodeV1 {
    /// Tree-sitter grammar kind.
    #[serde(rename = "k")]
    kind: String,
    /// Ordered child nodes and exact source text between them.
    #[serde(rename = "p")]
    parts: Vec<SyntaxPartV1>,
}

/// One ordered element inside a syntax node.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "t", deny_unknown_fields)]
enum SyntaxPartV1 {
    /// Child grammar node, optionally named by its parent field.
    #[serde(rename = "n")]
    Node {
        #[serde(rename = "f", skip_serializing_if = "Option::is_none", default)]
        field: Option<String>,
        #[serde(rename = "v")]
        value: Box<SyntaxNodeV1>,
    },
    /// Exact source bytes belonging to a leaf or grammar-trivia gap.
    #[serde(rename = "x")]
    Text {
        #[serde(rename = "v")]
        value: String,
    },
}

/// Failure to parse or validate a serialized syntax document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SyntaxError {
    /// Tree-sitter reported an error or inserted a missing node.
    InvalidZsh {
        /// Byte offset where the first error begins.
        byte: usize,
        /// Erroneous grammar node kind.
        kind: String,
    },
    /// A serialized document names a schema this crate does not understand.
    UnsupportedVersion(u8),
    /// The serialized tree's node structure does not match its rendered text.
    StructureMismatch,
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidZsh { byte, kind } => {
                write!(formatter, "invalid Zsh syntax near byte {byte} ({kind})")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported shell syntax schema version {version}")
            }
            Self::StructureMismatch => {
                formatter.write_str("serialized shell syntax structure does not match its rendered Zsh text")
            }
        }
    }
}

impl std::error::Error for SyntaxError {}

impl SyntaxDocumentV1 {
    /// Parse Zsh source into the lossless version-1 representation.
    pub fn parse(source: &str) -> Result<Self, SyntaxError> {
        let tree = parse_zsh(source)?;
        Ok(Self {
            version: SYNTAX_SCHEMA_VERSION,
            dialect: ShellDialect::Zsh,
            root: build_node(tree.root_node(), source),
        })
    }

    /// Return this document's schema version.
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// Return the shell dialect this document uses.
    pub const fn dialect(&self) -> ShellDialect {
        self.dialect
    }

    /// Reconstruct the exact source represented by this document.
    pub fn render(&self) -> String {
        let mut rendered = String::new();
        self.root.render_into(&mut rendered);
        rendered
    }

    /// Validate schema, rendered syntax, and the complete grammar structure.
    ///
    /// Requiring structural equality prevents a producer from attaching
    /// trusted-looking node names to different shell text.
    pub fn validate(&self) -> Result<(), SyntaxError> {
        if self.version != SYNTAX_SCHEMA_VERSION {
            return Err(SyntaxError::UnsupportedVersion(self.version));
        }
        let reparsed = Self::parse(&self.render())?;
        if reparsed.dialect != self.dialect || reparsed.root != self.root {
            return Err(SyntaxError::StructureMismatch);
        }
        Ok(())
    }
}

impl SyntaxNodeV1 {
    fn render_into(&self, output: &mut String) {
        for part in &self.parts {
            match part {
                SyntaxPartV1::Node { value, .. } => value.render_into(output),
                SyntaxPartV1::Text { value } => output.push_str(value),
            }
        }
    }
}

fn parser() -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_zsh::LANGUAGE.into())
        .expect("pinned tree-sitter and tree-sitter-zsh versions must be compatible");
    parser
}

fn parse_zsh(source: &str) -> Result<Tree, SyntaxError> {
    let tree = parser()
        .parse(source, None)
        .expect("parsing without a cancellation callback cannot be cancelled");
    if let Some(error) = first_error(tree.root_node()) {
        return Err(SyntaxError::InvalidZsh {
            byte: error.start_byte(),
            kind: error.kind().to_owned(),
        });
    }
    Ok(tree)
}

fn first_error(node: Node<'_>) -> Option<Node<'_>> {
    if node.is_error() || node.is_missing() {
        return Some(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.has_error()
            && let Some(error) = first_error(child)
        {
            return Some(error);
        }
    }
    None
}

fn build_node(node: Node<'_>, source: &str) -> SyntaxNodeV1 {
    let mut parts = Vec::new();
    let mut position = node.start_byte();
    let mut cursor = node.walk();
    for (index, child) in node.children(&mut cursor).enumerate() {
        if position < child.start_byte() {
            parts.push(SyntaxPartV1::Text {
                value: source[position..child.start_byte()].to_owned(),
            });
        }
        parts.push(SyntaxPartV1::Node {
            field: node.field_name_for_child(index as u32).map(str::to_owned),
            value: Box::new(build_node(child, source)),
        });
        position = child.end_byte();
    }
    if position < node.end_byte() {
        parts.push(SyntaxPartV1::Text {
            value: source[position..node.end_byte()].to_owned(),
        });
    }
    SyntaxNodeV1 {
        kind: node.kind().to_owned(),
        parts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZSH_PROGRAMS: &[&str] = &[
        "find /var/log -type f -mtime -2",
        "print -r -- **/*.rs(N.) | sort -u >| files.txt",
        "comm -13 <(sort old) <(sort new)",
        "typeset -a files=(**/*.rs(N.)); print -rC1 -- $files",
        "if [[ -n $name ]]; then print -r -- ${name:u}; else return 1; fi",
        "repeat 3 { print -r -- $RANDOM }",
        "cat <<'EOF'\nliteral $HOME\nEOF\n",
    ];

    #[test]
    fn losslessly_round_trips_representative_zsh() {
        for source in ZSH_PROGRAMS {
            let document = SyntaxDocumentV1::parse(source).unwrap_or_else(|error| {
                panic!("failed to parse {source:?}: {error}");
            });
            assert_eq!(document.version(), SYNTAX_SCHEMA_VERSION);
            assert_eq!(document.dialect(), ShellDialect::Zsh);
            assert_eq!(document.render(), *source);
            document.validate().unwrap();
        }
    }

    #[test]
    fn serde_round_trip_preserves_contract() {
        let document = SyntaxDocumentV1::parse(ZSH_PROGRAMS[1]).unwrap();
        let json = serde_json::to_string(&document).unwrap();
        let decoded: SyntaxDocumentV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, document);
        decoded.validate().unwrap();
    }

    #[test]
    fn serialized_envelope_uses_versioned_compact_keys() {
        let document = SyntaxDocumentV1::parse("print ok").unwrap();
        let value = serde_json::to_value(document).unwrap();
        let object = value.as_object().unwrap();
        assert_eq!(object.len(), 3);
        assert_eq!(object["v"], SYNTAX_SCHEMA_VERSION);
        assert_eq!(object["d"], "zsh");
        assert!(object.contains_key("r"));
    }

    #[test]
    fn rejects_invalid_zsh() {
        let error = SyntaxDocumentV1::parse("print 'unterminated").unwrap_err();
        assert!(matches!(error, SyntaxError::InvalidZsh { .. }));
    }

    #[test]
    fn validation_rejects_unknown_schema() {
        let mut document = SyntaxDocumentV1::parse("print ok").unwrap();
        document.version = 2;
        assert_eq!(document.validate(), Err(SyntaxError::UnsupportedVersion(2)));
    }

    #[test]
    fn validation_rejects_forged_structure() {
        let mut document = SyntaxDocumentV1::parse("print ok").unwrap();
        document.root.kind = "for_statement".to_owned();
        assert_eq!(document.validate(), Err(SyntaxError::StructureMismatch));
    }
}
