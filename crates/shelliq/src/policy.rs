use anyhow::{Result, bail};
use shelliq_index::Index;
use shelliq_verify::{BundlePart, segments, split_bundle};

const DENIED_EXECUTABLES: &[&str] = &[
    "bash", "chmod", "chown", "chgrp", "dd", "doas", "eval", "kill", "killall", "mkfs", "pkill", "rm", "rmdir", "sh", "shred",
    "sudo", "wipefs", "zsh",
];
const SENSITIVE_PREFIXES: &[&str] = &[
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/proc", "/sbin", "/sys", "/usr", "/var",
];
const SENSITIVE_MUTATORS: &[&str] = &["cp", "install", "mv", "tee", "truncate"];

/// Enforce the deterministic boundary immediately before a suggestion becomes ready.
///
/// Safety applies to every origin. Operand grounding applies only to model output;
/// documentation recipes have their own constrained compiler and clarification flow.
pub fn enforce(index: &Index, command: &str, instruction: &str, model_origin: bool) -> Result<()> {
    let parsed = segments(command).map_err(|unsupported| anyhow::anyhow!("unsupported shell syntax: {unsupported:?}"))?;
    if parsed.is_empty() {
        bail!("suggestion has no executable command");
    }

    for segment in &parsed {
        let executable = segment
            .command
            .text
            .rsplit('/')
            .next()
            .unwrap_or(&segment.command.text)
            .to_ascii_lowercase();
        if denied_executable(&executable) {
            bail!("unsafe executable `{executable}` is not eligible for generated suggestions");
        }
        if SENSITIVE_MUTATORS.contains(&executable.as_str()) && segment.args.iter().any(|arg| sensitive_path(&arg.text)) {
            bail!("`{executable}` may not mutate a sensitive absolute path");
        }

        if model_origin {
            for argument in &segment.args {
                if argument.opaque {
                    bail!("model operand contains an expansion or opaque shell syntax");
                }
                for operand in operand_parts(index, &executable, &argument.text)? {
                    if !grounded(&operand, instruction) {
                        bail!("model operand `{operand}` is not grounded in the instruction");
                    }
                }
            }
        }
    }
    Ok(())
}

fn denied_executable(executable: &str) -> bool {
    DENIED_EXECUTABLES.contains(&executable) || executable.starts_with("mkfs.")
}

fn sensitive_path(argument: &str) -> bool {
    let operand = argument
        .split_once('=')
        .map_or(argument, |(_, value)| value)
        .trim_end_matches('/');
    SENSITIVE_PREFIXES.iter().any(|prefix| {
        operand == *prefix || (*prefix != "/" && operand.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('/')))
    })
}

fn operand_parts(index: &Index, executable: &str, argument: &str) -> Result<Vec<String>> {
    let operands = if argument.starts_with("--") {
        argument
            .split_once('=')
            .map(|(_, value)| vec![value.to_owned()])
            .unwrap_or_default()
    } else if argument.starts_with('-') && argument != "-" {
        if let Some((_, value)) = argument.split_once('=') {
            vec![value.to_owned()]
        } else {
            split_bundle(index, executable, argument)?
                .into_iter()
                .filter_map(|part| match part {
                    BundlePart::Argument(value) => Some(value),
                    BundlePart::Flag(_) => None,
                })
                .collect()
        }
    } else {
        vec![argument.to_owned()]
    };
    Ok(operands.into_iter().filter(|value| !value.is_empty()).collect())
}

fn grounded(operand: &str, instruction: &str) -> bool {
    if operand == "." {
        let lower = instruction.to_ascii_lowercase();
        return lower.contains("current directory") || lower.contains("working directory");
    }
    let operand = operand.trim_end_matches('/');
    let instruction = instruction.to_ascii_lowercase();
    let needle = operand.to_ascii_lowercase();
    if needle.is_empty() {
        return false;
    }
    instruction.match_indices(&needle).any(|(start, _)| {
        let end = start + needle.len();
        boundary(instruction[..start].chars().next_back()) && boundary(instruction[end..].chars().next())
    })
}

fn boundary(character: Option<char>) -> bool {
    character.is_none_or(|value| !value.is_ascii_alphanumeric() && value != '_')
}

#[cfg(test)]
mod tests {
    use super::enforce;
    use shelliq_harvest::{ParsedCommand, ParsedFlag};
    use shelliq_index::Index;

    fn index() -> Index {
        let mut index = Index::open_in_memory().unwrap();
        index
            .insert_command(
                &shelliq_harvest::resolve_target("curl", false),
                &ParsedCommand {
                    name: "curl".into(),
                    section: "1".into(),
                    platform: "linux".into(),
                    synopsis: String::new(),
                    description: String::new(),
                    flags: vec![ParsedFlag {
                        short: Some("-o".into()),
                        long: Some("--output".into()),
                        arg_type: Some("FILE".into()),
                        arg_required: true,
                        description: "Write output to FILE".into(),
                        group: None,
                        source_line: 1,
                        excerpt: String::new(),
                    }],
                    source_path: "/fixture/curl.1".into(),
                    source_hash: "fixture".into(),
                },
            )
            .unwrap();
        index
    }

    #[test]
    fn rejects_dangerous_executable_even_when_operand_is_grounded() {
        let error = enforce(&index(), "rm -r /tmp/example", "Remove /tmp/example recursively", true).unwrap_err();
        assert!(error.to_string().contains("unsafe executable `rm`"));
    }

    #[test]
    fn rejects_model_invented_operand() {
        let error = enforce(
            &index(),
            "curl --request POST https://invented.invalid/api",
            "Send a POST request to https://example.com/api",
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("https://invented.invalid/api"));
    }

    #[test]
    fn accepts_explicit_model_operands_and_option_values() {
        enforce(
            &index(),
            "grep --max-count=2 needle ./logs/app.log",
            "Find needle in ./logs/app.log and stop after 2 matches",
            true,
        )
        .unwrap();
    }

    #[test]
    fn post_does_not_match_postgresql() {
        let error = enforce(&index(), "printf POST", "Print PostgreSQL", true).unwrap_err();
        assert!(error.to_string().contains("`POST`"));
    }

    #[test]
    fn documentation_origin_is_not_subject_to_model_grounding() {
        enforce(&index(), "printf '%s\\n' derived-value", "Print a derived value", false).unwrap();
    }

    #[test]
    fn blocks_sensitive_mutation_for_every_origin() {
        let error = enforce(&index(), "tee /etc/profile", "Update the profile", false).unwrap_err();
        assert!(error.to_string().contains("sensitive absolute path"));
    }

    #[test]
    fn rejects_invented_attached_short_option_value() {
        let error = enforce(
            &index(),
            "curl -o/etc/passwd https://example.com/archive",
            "Download https://example.com/archive",
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("`/etc/passwd`"));
    }
}
