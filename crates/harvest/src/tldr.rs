//! tldr-pages ingestion — the task-language vocabulary bridge for search.
//!
//! man pages describe mechanism ("moved to a different location"); tldr describes task
//! ("follow redirects"). See `PLAN.md` section 4. Pages are parsed out of a vendored archive
//! embedded in the binary at compile time, so ingestion needs no `tldr` install and no
//! network access at runtime. `crates/harvest/vendor/NOTICE.md` carries the CC-BY-4.0
//! attribution for the vendored content.
//!
//! This module only reports what a page says. Whether a flag it mentions actually exists on
//! this machine's target is decided by the index, not here — tldr pages are generic, and a
//! GNU-only flag must never become a fact on a Mac just because a page mentions it.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use zip::ZipArchive;

/// Bumped whenever tldr parsing behaviour changes, invalidating previously ingested examples
/// the way `PARSER_VERSION` does for man pages.
pub const TLDR_PARSER_VERSION: u32 = 1;

static ARCHIVE: &[u8] = include_bytes!("../vendor/tldr-pages.en.zip");

/// One example as tldr writes it: a task-phrased description, the command with its
/// placeholders rendered for display, and the flag spellings mentioned in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedExample {
    pub description: String,
    pub text: String,
    /// Original command recipe, including TLDR placeholder markup.
    pub template: String,
    pub flags: Vec<String>,
}

/// tldr's platform directory for shelliq's `platform()` string.
fn tldr_platform(platform: &str) -> &'static str {
    match platform {
        "darwin" => "osx",
        _ => "linux",
    }
}

/// Examples for `name`, preferring the platform-specific page and falling back to `common`,
/// the way tldr itself resolves a page. Absent from the archive is not an error: most
/// commands simply have no tldr page.
pub fn harvest_tldr(name: &str, platform: &str) -> Result<Vec<ParsedExample>> {
    let mut archive = ZipArchive::new(Cursor::new(ARCHIVE)).context("opening vendored tldr archive")?;
    let page = read_page(&mut archive, &format!("{}/{name}.md", tldr_platform(platform)))
        .or_else(|| read_page(&mut archive, &format!("common/{name}.md")));
    Ok(page.map(|text| parse_page(&text)).unwrap_or_default())
}

/// Examples for an executable and TLDR's separately filed subcommand pages.
///
/// Pages such as `aws-configure.md` and `cargo-add.md` document commands whose
/// executable is `aws` or `cargo`. Selection remains generic: the filename must
/// be the executable name or begin with `<name>-`; the compiler separately
/// verifies the recipe's first command word.
pub fn harvest_tldr_family(name: &str, platform: &str) -> Result<Vec<ParsedExample>> {
    let mut archive = ZipArchive::new(Cursor::new(ARCHIVE)).context("opening vendored tldr archive")?;
    let preferred = tldr_platform(platform);
    let mut pages = BTreeMap::<String, (bool, String)>::new();
    for index in 0..archive.len() {
        let mut file = archive.by_index(index).context("reading vendored tldr member")?;
        let path = file.name().to_string();
        let Some((directory, filename)) = path.split_once('/') else {
            continue;
        };
        if directory != preferred && directory != "common" {
            continue;
        }
        let Some(stem) = filename.strip_suffix(".md") else {
            continue;
        };
        if stem != name && !stem.strip_prefix(name).is_some_and(|suffix| suffix.starts_with('-')) {
            continue;
        }
        let is_preferred = directory == preferred;
        if pages.get(stem).is_some_and(|(current, _)| *current && !is_preferred) {
            continue;
        }
        let mut text = String::new();
        file.read_to_string(&mut text).context("reading vendored tldr page")?;
        pages.insert(stem.to_string(), (is_preferred, text));
    }
    Ok(pages.into_values().flat_map(|(_, text)| parse_page(&text)).collect())
}

fn read_page(archive: &mut ZipArchive<Cursor<&[u8]>>, path: &str) -> Option<String> {
    let mut file = archive.by_name(path).ok()?;
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    Some(text)
}

/// Parse a tldr page's body: `- description:` bullets, each followed by a backtick-quoted
/// command line on the next non-blank line.
pub fn parse_page(text: &str) -> Vec<ParsedExample> {
    let lines: Vec<&str> = text.lines().collect();
    let mut examples = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        let Some(desc) = line.strip_prefix("- ") else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j < lines.len() && lines[j].trim().is_empty() {
            j += 1;
        }
        let Some(cmd_line) = lines.get(j).map(|l| l.trim()) else {
            i += 1;
            continue;
        };
        if cmd_line.len() < 2 || !cmd_line.starts_with('`') || !cmd_line.ends_with('`') {
            i += 1;
            continue;
        }
        let raw = &cmd_line[1..cmd_line.len() - 1];
        examples.push(ParsedExample {
            description: desc.trim_end_matches(':').trim().to_string(),
            text: render_placeholders(raw),
            template: raw.to_string(),
            flags: extract_flags(raw),
        });
        i = j + 1;
    }
    examples
}

/// Strip tldr's `{{...}}` placeholder markup for a human-readable command line. Not used for
/// flag validation — only for the free-text side of the search index.
fn render_placeholders(raw: &str) -> String {
    raw.replace("{{", "")
        .replace("}}", "")
        .replace(['[', ']'], "")
        .replace('|', " ")
}

/// Flag spellings mentioned in a raw command line, exactly as tldr writes them.
///
/// tldr marks flag placeholders explicitly, e.g. `{{[-L|--location]}}` or
/// `{{[-la|-l --all]}}`, which is a more reliable signal than scanning the rendered command
/// for dash-prefixed tokens — a bare `{{https://example.com}}` is an argument, not a flag.
/// Dash-prefixed tokens written literally outside any placeholder (`ls -1`) are still picked
/// up, since not every page bothers to mark up its flags.
fn extract_flags(raw: &str) -> Vec<String> {
    let mut flags = Vec::new();
    let mut literal = String::new();
    let mut idx = 0;
    while idx < raw.len() {
        if raw[idx..].starts_with("{{") {
            let rel_end = raw[idx..].find("}}");
            let end = rel_end.map(|e| idx + e).unwrap_or(raw.len());
            let inner = raw[idx + 2..end].trim();
            let body = inner.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(inner);
            for alt in body.split('|') {
                for token in alt.split_whitespace() {
                    if is_flag_token(token) {
                        flags.push(token.to_string());
                    }
                }
            }
            idx = if rel_end.is_some() { end + 2 } else { raw.len() };
        } else {
            let next = raw[idx..].find("{{").map(|p| idx + p).unwrap_or(raw.len());
            literal.push_str(&raw[idx..next]);
            literal.push(' ');
            idx = next;
        }
    }
    for token in literal.split_whitespace() {
        if is_flag_token(token) {
            flags.push(token.to_string());
        }
    }
    flags.sort();
    flags.dedup();
    flags
}

fn is_flag_token(token: &str) -> bool {
    let t = token.trim_end_matches([',', '.', ';', ':']);
    t.starts_with('-') && t.len() > 1 && t.chars().skip(1).all(|c| c.is_ascii_alphanumeric() || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bracketed_alternative_flags() {
        let raw = "curl {{[-L|--location]}} {{[-D|--dump-header]}} - {{https://example.com}}";
        let flags = extract_flags(raw);
        assert_eq!(flags, vec!["--dump-header", "--location", "-D", "-L"]);
    }

    #[test]
    fn parses_short_and_long_combined_alternative() {
        let flags = extract_flags("ls {{[-la|-l --all]}}");
        assert_eq!(flags, vec!["--all", "-l", "-la"]);
    }

    #[test]
    fn picks_up_literal_flags_outside_placeholders() {
        let flags = extract_flags("ls -1");
        assert_eq!(flags, vec!["-1"]);
    }

    #[test]
    fn placeholders_that_are_arguments_are_not_flags() {
        let flags = extract_flags("curl {{https://example.com}}");
        assert!(flags.is_empty());
    }

    #[test]
    fn parses_a_full_page_into_examples() {
        let page = "# curl\n\n\
                     > Transfers data from or to a server.\n\n\
                     - Make an HTTP GET request and dump the contents in `stdout`:\n\n\
                     `curl {{https://example.com}}`\n\n\
                     - Follow redirects and dump the reply headers:\n\n\
                     `curl {{[-L|--location]}} {{https://example.com}}`\n";
        let examples = parse_page(page);
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[1].description, "Follow redirects and dump the reply headers");
        assert_eq!(examples[1].flags, vec!["--location", "-L"]);
        assert_eq!(examples[1].template, "curl {{[-L|--location]}} {{https://example.com}}");
        assert!(examples[1].text.contains("curl"));
    }

    #[test]
    fn real_curl_page_lists_location_among_its_flags() {
        let examples = harvest_tldr("curl", "linux").unwrap();
        assert!(
            examples
                .iter()
                .any(|e| e.flags.iter().any(|f| f == "-L" || f == "--location")),
            "expected at least one curl example mentioning -L/--location"
        );
    }

    #[test]
    fn falls_back_to_common_when_no_platform_page_exists() {
        let examples = harvest_tldr("grep", "linux").unwrap();
        assert!(!examples.is_empty(), "grep should have a common tldr page");
    }

    #[test]
    fn unknown_command_yields_no_examples_rather_than_an_error() {
        let examples = harvest_tldr("definitely-not-a-real-command-xyz", "linux").unwrap();
        assert!(examples.is_empty());
    }

    #[test]
    fn family_harvest_includes_separately_filed_subcommands() {
        let examples = harvest_tldr_family("aws", "linux").unwrap();
        assert!(
            examples
                .iter()
                .any(|example| { example.template.starts_with("aws configure") && example.template.contains("--profile") }),
            "expected aws-configure page in aws family"
        );
    }
}
