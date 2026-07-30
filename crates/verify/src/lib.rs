//! Checking a command line against the index.
//!
//! The verifier is what makes a generated command trustworthy: every flag is confirmed
//! against the man page installed on this machine, and anything that cannot be confirmed
//! is labelled rather than waved through.
//!
//! Bundled short flags are the reason this cannot be a naive token scan. Real command
//! lines are full of `-fsSL`, `-sirn`, `-xzf`, and `-LsSf`, and a wrong case hides
//! invisibly inside a bundle — `-sirN` looks fine at a glance. Decomposing bundles is
//! therefore a correctness requirement, not a nicety.

use anyhow::Result;
use shelliq_index::{FlagLookup, FlagRow, Index};

/// One `cmd ... ` stage of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub command: String,
    pub args: Vec<String>,
}

/// What the verifier concluded about one flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// Confirmed against the index, with a citation.
    Verified { token: String, flag: Box<FlagRow> },
    /// Exists under a different case. The answer to the original complaint.
    WrongCase { token: String, suggestion: Box<FlagRow> },
    /// No such flag. Never treated as acceptable.
    UnknownFlag { command: String, token: String },
    /// The command itself is not indexed, so nothing about it can be asserted.
    UnknownCommand { command: String },
    /// A flag that requires an argument did not get one.
    MissingArgument { token: String, flag: Box<FlagRow> },
}

impl Finding {
    /// Whether this finding is consistent with a correct command line.
    pub fn is_clean(&self) -> bool {
        matches!(self, Finding::Verified { .. })
    }
}

/// Split a command line into pipeline segments.
///
/// Uses `shlex` so quoting is respected: `grep -r "a | b"` is one segment, not two.
pub fn segments(line: &str) -> Vec<Segment> {
    let Some(tokens) = shlex::split(line) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for token in tokens {
        if matches!(token.as_str(), "|" | "||" | "&&" | ";" | "&") {
            push_segment(&mut out, &mut current);
        } else {
            current.push(token);
        }
    }
    push_segment(&mut out, &mut current);
    out
}

fn push_segment(out: &mut Vec<Segment>, current: &mut Vec<String>) {
    if current.is_empty() {
        return;
    }
    let command = current.remove(0);
    out.push(Segment {
        command,
        args: std::mem::take(current),
    });
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

/// Verify every flag in a command line.
pub fn verify(index: &Index, line: &str) -> Result<Vec<Finding>> {
    let mut findings = Vec::new();

    for segment in segments(line) {
        if !index.command_exists(&segment.command)? {
            findings.push(Finding::UnknownCommand {
                command: segment.command.clone(),
            });
            continue;
        }

        let mut args = segment.args.iter().peekable();
        while let Some(arg) = args.next() {
            if arg == "--" {
                break;
            }
            if !arg.starts_with('-') || arg == "-" {
                continue;
            }

            // A long flag may carry its argument inline: `--block-size=M`.
            let (token, inline_arg) = match arg.split_once('=') {
                Some((t, v)) if t.starts_with("--") => (t.to_string(), Some(v.to_string())),
                _ => (arg.clone(), None),
            };

            let parts = split_bundle(index, &segment.command, &token)?;
            let mut consumed_inline = inline_arg.is_some();

            for (i, part) in parts.iter().enumerate() {
                let BundlePart::Flag(flag_token) = part else {
                    consumed_inline = true;
                    continue;
                };
                let has_attached_arg = matches!(parts.get(i + 1), Some(BundlePart::Argument(_)));

                match index.lookup_flag(&segment.command, flag_token)? {
                    FlagLookup::Exact(flag) => {
                        let needs_arg = flag.arg_type.is_some() && flag.arg_required;
                        let satisfied = has_attached_arg
                            || consumed_inline
                            || args.peek().is_some_and(|n| !n.starts_with('-'));
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
                            command: segment.command.clone(),
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
        }
    }

    fn seeded() -> Index {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_command(&ParsedCommand {
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
        })
        .unwrap();
        idx.insert_command(&ParsedCommand {
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
        })
        .unwrap();
        idx
    }

    #[test]
    fn pipeline_splits_into_segments() {
        let segs = segments("grep -r foo | head -5");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].command, "grep");
        assert_eq!(segs[1].command, "head");
    }

    #[test]
    fn quoted_pipe_is_not_a_segment_boundary() {
        let segs = segments(r#"grep -r "a | b" ."#);
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
        assert_eq!(
            parts,
            vec![
                BundlePart::Flag("-A".into()),
                BundlePart::Argument("5".into()),
            ]
        );
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
                Finding::WrongCase { token, suggestion } => {
                    Some((token.clone(), suggestion.short.clone().unwrap()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(wrong, vec![("-N".to_string(), "-n".to_string())]);
    }

    #[test]
    fn a_correct_line_is_entirely_clean() {
        let idx = seeded();
        let findings = verify(&idx, "tar -xzf archive.tar.gz").unwrap();
        assert!(
            findings.iter().all(Finding::is_clean),
            "expected all clean, got {findings:?}"
        );
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
}
