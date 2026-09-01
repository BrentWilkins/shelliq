//! Checking a command line against the index.
//!
//! The claim this makes is narrow and worth stating exactly: each literal option spelling
//! in the line exists for that command in the index built from this machine's man pages,
//! with roughly the right argument shape. Not that the command is correct, that its
//! operands are right, that the flags make sense together, or that running it is safe. See
//! the claim ladder in `PLAN.md`.
//!
//! Bundled short flags are the reason this cannot be a naive token scan. Real command
//! lines are full of `-fsSL`, `-sirn`, `-xzf`, and `-LsSf`, and a wrong case hides
//! invisibly inside a bundle — `-sirN` looks fine at a glance. Decomposing bundles is
//! therefore a correctness requirement, not a nicety.
//!
//! The other half of honesty is refusing to answer. Syntax the scanner cannot analyse
//! produces a [`Finding::Unsupported`] abstention and a non-zero exit, never silence.

mod syntax;

use anyhow::Result;
use shelliq_index::{FlagLookup, FlagRow, Index};

pub use syntax::{Segment, Unsupported, Word, segments};

/// What the checker concluded about one flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// Found in the index, with a citation.
    Verified { token: String, flag: Box<FlagRow> },
    /// Exists under a different case. The answer to the original complaint.
    WrongCase { token: String, suggestion: Box<FlagRow> },
    /// No such flag. Never treated as acceptable.
    UnknownFlag { command: String, token: String },
    /// The command itself is not indexed, so nothing about it can be asserted.
    UnknownCommand { command: String },
    /// A flag that requires an argument did not get one.
    MissingArgument { token: String, flag: Box<FlagRow> },
    /// Shell syntax the checker does not analyse. Reported explicitly, because the
    /// alternative — saying nothing — reads as approval.
    Unsupported { construct: Unsupported, text: String },
}

impl Finding {
    /// Whether this finding is consistent with a correct command line.
    ///
    /// An abstention is not clean. Nothing was checked, so nothing can be claimed.
    pub fn is_clean(&self) -> bool {
        matches!(self, Finding::Verified { .. })
    }
}

/// One element of a decomposed bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundlePart {
    /// A single short flag, e.g. `-s` from `-sirn`.
    Flag(String),
    /// Text consumed as a flag's attached argument, e.g. `5` in `-A5`.
    Argument(String),
}

/// Decompose a bundled short-flag token.
///
/// `-sirn` becomes `-s -i -r -n`. Splitting stops when a flag takes an argument, because
/// the rest of the token is that argument rather than more flags: `grep -A5` is `-A` with
/// `5`, not `-A -5`. The index decides which case applies, so this is not guesswork.
///
/// Long flags and bare `-`/`--` are returned unchanged as a single part.
pub fn split_bundle(index: &Index, command: &str, token: &str) -> Result<Vec<BundlePart>> {
    if !token.starts_with('-') || token.starts_with("--") || token.len() <= 2 {
        return Ok(vec![BundlePart::Flag(token.to_string())]);
    }
    // Some utilities use multi-character single-dash options (`find -type`, Java
    // `-classpath`). An exact local fact takes precedence over interpreting the token as
    // a POSIX-style short-option bundle.
    if matches!(index.lookup_flag(command, token)?, FlagLookup::Exact(_)) {
        return Ok(vec![BundlePart::Flag(token.to_string())]);
    }

    let chars: Vec<char> = token[1..].chars().collect();
    let mut parts = Vec::new();

    for (i, c) in chars.iter().enumerate() {
        let candidate = format!("-{c}");
        let lookup = index.lookup_flag(command, &candidate)?;

        let takes_arg = match &lookup {
            FlagLookup::Exact(f) => f.arg_type.is_some() && f.arg_required,
            _ => false,
        };

        parts.push(BundlePart::Flag(candidate));

        if takes_arg && i + 1 < chars.len() {
            let rest: String = chars[i + 1..].iter().collect();
            parts.push(BundlePart::Argument(rest));
            break;
        }
    }

    Ok(parts)
}

/// Check every flag in a command line.
pub fn verify(index: &Index, line: &str) -> Result<Vec<Finding>> {
    let mut findings = Vec::new();

    // Fail closed. A line the scanner cannot read yields one abstention rather than an
    // empty result: zero findings is indistinguishable from a clean line to every caller,
    // and to the exit status, which is how P0 came to approve `grep -r "unclosed`.
    let segments = match segments(line) {
        Ok(segments) => segments,
        Err(construct) => {
            return Ok(vec![Finding::Unsupported {
                construct,
                text: line.to_string(),
            }]);
        }
    };

    for segment in segments {
        // An expanded command name could be anything, so nothing downstream is checkable.
        if segment.command.opaque {
            findings.push(Finding::Unsupported {
                construct: Unsupported::Expansion,
                text: segment.command.text.clone(),
            });
            continue;
        }
        if !index.command_exists(&segment.command.text)? {
            findings.push(Finding::UnknownCommand {
                command: segment.command.text.clone(),
            });
            continue;
        }

        let mut args = segment.args.iter().peekable();
        while let Some(arg) = args.next() {
            // `$OPTS` may expand to flags, so it cannot be dismissed as an operand. The
            // rest of the line is still worth checking, so this costs only the one word.
            if arg.opaque {
                findings.push(Finding::Unsupported {
                    construct: Unsupported::Expansion,
                    text: arg.text.clone(),
                });
                continue;
            }
            if arg.text == "--" {
                break;
            }
            if !arg.text.starts_with('-') || arg.text == "-" {
                continue;
            }

            // Long flags and exact multi-character single-dash flags may carry
            // their arguments inline: `--block-size=M`, `-type=AXFR`.
            let inline = arg.text.split_once('=');
            let exact_single_dash_inline = if let Some((candidate, _)) = inline {
                candidate.starts_with('-')
                    && !candidate.starts_with("--")
                    && candidate.len() > 2
                    && matches!(index.lookup_flag(&segment.command.text, candidate)?, FlagLookup::Exact(_))
            } else {
                false
            };
            let (token, inline_arg) = match inline {
                Some((t, v)) if t.starts_with("--") || exact_single_dash_inline => (t.to_string(), Some(v.to_string())),
                _ => (arg.text.clone(), None),
            };

            let parts = split_bundle(index, &segment.command.text, &token)?;
            let mut consumed_inline = inline_arg.is_some();

            for (i, part) in parts.iter().enumerate() {
                let BundlePart::Flag(flag_token) = part else {
                    consumed_inline = true;
                    continue;
                };
                let has_attached_arg = matches!(parts.get(i + 1), Some(BundlePart::Argument(_)));

                match index.lookup_flag(&segment.command.text, flag_token)? {
                    FlagLookup::Exact(flag) => {
                        let needs_arg = flag.arg_type.is_some() && flag.arg_required;
                        let satisfied =
                            has_attached_arg || consumed_inline || args.peek().is_some_and(|n| !n.text.starts_with('-'));
                        if needs_arg && !satisfied {
                            findings.push(Finding::MissingArgument {
                                token: flag_token.clone(),
                                flag,
                            });
                        } else {
                            if needs_arg && !has_attached_arg && !consumed_inline {
                                args.next();
                            }
                            findings.push(Finding::Verified {
                                token: flag_token.clone(),
                                flag,
                            });
                        }
                    }
                    FlagLookup::CaseMismatch { suggestion, .. } => {
                        findings.push(Finding::WrongCase {
                            token: flag_token.clone(),
                            suggestion,
                        });
                    }
                    FlagLookup::Unknown { .. } => {
                        findings.push(Finding::UnknownFlag {
                            command: segment.command.text.clone(),
                            token: flag_token.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shelliq_harvest::{ParsedCommand, ParsedFlag};

    fn f(short: &str, long: &str, arg: Option<&str>, line: usize) -> ParsedFlag {
        ParsedFlag {
            short: Some(short.to_string()),
            long: Some(long.to_string()),
            arg_type: arg.map(str::to_string),
            arg_required: arg.is_some(),
            description: format!("description of {short}"),
            group: None,
            source_line: line,
            excerpt: String::new(),
        }
    }

    fn seeded() -> Index {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_command(
            &shelliq_harvest::resolve_target("grep", false),
            &ParsedCommand {
                name: "grep".into(),
                section: "1".into(),
                platform: "linux".into(),
                synopsis: String::new(),
                description: String::new(),
                source_path: String::new(),
                source_hash: String::new(),
                flags: vec![
                    f("-s", "--no-messages", None, 10),
                    f("-i", "--ignore-case", None, 45),
                    f("-r", "--recursive", None, 168),
                    f("-R", "--dereference-recursive", None, 171),
                    f("-n", "--line-number", None, 101),
                    f("-A", "--after-context", Some("NUM"), 60),
                ],
            },
        )
        .unwrap();
        idx.insert_command(
            &shelliq_harvest::resolve_target("tar", false),
            &ParsedCommand {
                name: "tar".into(),
                section: "1".into(),
                platform: "linux".into(),
                synopsis: String::new(),
                description: String::new(),
                source_path: String::new(),
                source_hash: String::new(),
                flags: vec![
                    f("-c", "--create", None, 20),
                    f("-z", "--gzip", None, 30),
                    f("-f", "--file", Some("ARCHIVE"), 40),
                    f("-x", "--extract", None, 50),
                ],
            },
        )
        .unwrap();
        idx
    }

    #[test]
    fn pipeline_splits_into_segments() {
        let segs = segments("grep -r foo | head -5").unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].command.text, "grep");
        assert_eq!(segs[1].command.text, "head");
    }

    #[test]
    fn quoted_pipe_is_not_a_segment_boundary() {
        let segs = segments(r#"grep -r "a | b" ."#).unwrap();
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn bundle_decomposes_into_single_flags() {
        let idx = seeded();
        let parts = split_bundle(&idx, "grep", "-sirn").unwrap();
        assert_eq!(
            parts,
            vec![
                BundlePart::Flag("-s".into()),
                BundlePart::Flag("-i".into()),
                BundlePart::Flag("-r".into()),
                BundlePart::Flag("-n".into()),
            ]
        );
    }

    #[test]
    fn bundle_stops_at_a_flag_that_takes_an_argument() {
        let idx = seeded();
        // -A takes NUM, so `5` is its argument, not the flag `-5`.
        let parts = split_bundle(&idx, "grep", "-A5").unwrap();
        assert_eq!(parts, vec![BundlePart::Flag("-A".into()), BundlePart::Argument("5".into()),]);
    }

    #[test]
    fn exact_multi_character_single_dash_option_is_not_split_as_a_bundle() {
        let mut idx = seeded();
        idx.insert_command(
            &shelliq_harvest::resolve_target("find", false),
            &ParsedCommand {
                name: "find".into(),
                section: "1".into(),
                platform: std::env::consts::OS.into(),
                synopsis: String::new(),
                description: String::new(),
                source_path: String::new(),
                source_hash: String::new(),
                flags: vec![f("-type", "", Some("c"), 1)],
            },
        )
        .unwrap();

        assert_eq!(
            split_bundle(&idx, "find", "-type").unwrap(),
            vec![BundlePart::Flag("-type".into())]
        );

        let findings = verify(&idx, "find -type=f .").unwrap();
        assert_eq!(findings.len(), 1);
        assert!(findings.iter().all(Finding::is_clean), "got {findings:?}");
    }

    #[test]
    fn unknown_or_wrong_case_single_dash_inline_option_is_not_accepted() {
        let mut idx = seeded();
        idx.insert_command(
            &shelliq_harvest::resolve_target("find", false),
            &ParsedCommand {
                name: "find".into(),
                section: "1".into(),
                platform: std::env::consts::OS.into(),
                synopsis: String::new(),
                description: String::new(),
                source_path: String::new(),
                source_hash: String::new(),
                flags: vec![f("-type", "", Some("c"), 1)],
            },
        )
        .unwrap();

        for line in ["find -Type=f .", "find -bogus=f ."] {
            let findings = verify(&idx, line).unwrap();
            assert!(
                findings.iter().any(|finding| !finding.is_clean()),
                "unexpectedly accepted {line:?}: {findings:?}"
            );
        }
    }

    #[test]
    fn tar_style_bundle_with_trailing_file_flag() {
        let idx = seeded();
        let parts = split_bundle(&idx, "tar", "-xzf").unwrap();
        assert_eq!(
            parts,
            vec![
                BundlePart::Flag("-x".into()),
                BundlePart::Flag("-z".into()),
                BundlePart::Flag("-f".into()),
            ]
        );
    }

    /// The failure mode that motivated the whole project: a wrong case buried in a bundle.
    #[test]
    fn wrong_case_inside_a_bundle_is_caught() {
        let idx = seeded();
        let findings = verify(&idx, "grep -sirN pattern .").unwrap();
        let wrong: Vec<_> = findings
            .iter()
            .filter_map(|f| match f {
                Finding::WrongCase { token, suggestion } => Some((token.clone(), suggestion.short.clone().unwrap())),
                _ => None,
            })
            .collect();
        assert_eq!(wrong, vec![("-N".to_string(), "-n".to_string())]);
    }

    #[test]
    fn a_correct_line_is_entirely_clean() {
        let idx = seeded();
        let findings = verify(&idx, "tar -xzf archive.tar.gz").unwrap();
        assert!(findings.iter().all(Finding::is_clean), "expected all clean, got {findings:?}");
        assert_eq!(findings.len(), 3);
    }

    #[test]
    fn fabricated_flag_is_never_silently_accepted() {
        let idx = seeded();
        let findings = verify(&idx, "grep --frobnicate .").unwrap();
        assert!(matches!(
            findings.as_slice(),
            [Finding::UnknownFlag { token, .. }] if token == "--frobnicate"
        ));
    }

    #[test]
    fn unindexed_command_is_reported_not_assumed() {
        let idx = seeded();
        let findings = verify(&idx, "definitely-not-installed -r").unwrap();
        assert!(matches!(
            findings.as_slice(),
            [Finding::UnknownCommand { command }] if command == "definitely-not-installed"
        ));
    }

    #[test]
    fn flag_needing_an_argument_reports_when_it_lacks_one() {
        let idx = seeded();
        let findings = verify(&idx, "grep -A").unwrap();
        assert!(
            matches!(findings.as_slice(), [Finding::MissingArgument { token, .. }] if token == "-A"),
            "got {findings:?}"
        );
    }

    #[test]
    fn long_flag_with_inline_argument_is_accepted() {
        let idx = seeded();
        let findings = verify(&idx, "grep --after-context=3 x").unwrap();
        assert!(findings.iter().all(Finding::is_clean), "got {findings:?}");
    }

    #[test]
    fn separator_ends_flag_parsing() {
        let idx = seeded();
        // After `--`, `-r` is a filename, not a flag.
        let findings = verify(&idx, "grep -i -- -r").unwrap();
        assert_eq!(findings.len(), 1);
        assert!(findings[0].is_clean());
    }

    #[test]
    fn each_segment_of_a_pipeline_is_checked() {
        let idx = seeded();
        let findings = verify(&idx, "grep -r x | tar -xzf y").unwrap();
        assert_eq!(findings.len(), 4);
        assert!(findings.iter().all(Finding::is_clean));
    }

    /// The fail-open bug this phase exists to close. P0 returned zero findings here, which
    /// every caller — including the exit status — read as "clean".
    #[test]
    fn an_unbalanced_quote_abstains_instead_of_passing() {
        let idx = seeded();
        let findings = verify(&idx, r#"grep -r "unclosed"#).unwrap();
        let [Finding::Unsupported { construct, .. }] = findings.as_slice() else {
            panic!("expected one abstention, got {findings:?}");
        };
        assert_eq!(*construct, Unsupported::UnbalancedQuote);
        assert!(!findings[0].is_clean());
    }

    #[test]
    fn unreadable_syntax_abstains_for_the_whole_line() {
        let idx = seeded();
        for (line, expected) in [
            ("grep -r foo>out", Unsupported::Redirection),
            ("grep -r foo 2>/dev/null", Unsupported::Redirection),
            ("grep -r $(cat f)", Unsupported::CommandSubstitution),
            ("grep -r `cat f`", Unsupported::CommandSubstitution),
            ("(grep -r foo)", Unsupported::Subshell),
            (r"grep -r \", Unsupported::DanglingEscape),
        ] {
            let findings = verify(&idx, line).unwrap();
            assert!(
                matches!(
                    findings.as_slice(),
                    [Finding::Unsupported { construct, .. }] if *construct == expected
                ),
                "line {line:?} got {findings:?}"
            );
        }
    }

    /// Abstaining on the whole line for a redirection would be over-broad here: the flag
    /// is quoted text, not syntax, and P0's own test asserted the quoted case works.
    #[test]
    fn a_quoted_metacharacter_does_not_trigger_abstention() {
        let idx = seeded();
        let findings = verify(&idx, r#"grep -r "a > b" ."#).unwrap();
        assert!(findings.iter().all(Finding::is_clean), "got {findings:?}");
    }

    /// An expansion could be anything, including flags, so the word is not silently taken
    /// for an operand — but the flags either side of it are still worth reporting.
    #[test]
    fn an_expansion_costs_its_word_and_no_more() {
        let idx = seeded();
        let findings = verify(&idx, "grep -i $OPTS -r .").unwrap();
        assert_eq!(findings.len(), 3, "got {findings:?}");
        assert!(findings[0].is_clean());
        assert!(matches!(
            &findings[1],
            Finding::Unsupported { construct: Unsupported::Expansion, text } if text == "$OPTS"
        ));
        assert!(findings[2].is_clean());
    }

    #[test]
    fn an_expanded_command_name_is_not_looked_up() {
        let idx = seeded();
        let findings = verify(&idx, "$TOOL -r .").unwrap();
        let [Finding::Unsupported { construct, text }] = findings.as_slice() else {
            panic!("expected one abstention, got {findings:?}");
        };
        assert_eq!(*construct, Unsupported::Expansion);
        assert_eq!(text, "$TOOL");
    }

    /// P0 read `a|b` as one segment and so never checked `tar`.
    #[test]
    fn an_attached_pipe_still_checks_both_segments() {
        let idx = seeded();
        let findings = verify(&idx, "grep -r x|tar -xzf y").unwrap();
        assert_eq!(findings.len(), 4, "got {findings:?}");
        assert!(findings.iter().all(Finding::is_clean));
    }
}
