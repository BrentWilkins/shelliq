//! Harvesting command facts from man pages.
//!
//! P0 parses *rendered* `man` output rather than roff source. `mandoc -T markdown` is the
//! preferred path described in PLAN.md — it understands both `man` and `mdoc` macros, so
//! BSD and macOS pages work identically — but mandoc is not installed everywhere, so the
//! rendered path is what ships first. Both paths must agree on the same fixtures.
//!
//! GNU man-db and BSD mandoc render roff `.TP` and mdoc `.It` entries at different
//! margins. The parser discovers each section's option-tag margin from its structure;
//! `MANWIDTH` merely reduces wrapping and is not part of the grammar.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::process::Command;

pub mod help_crawler;
pub mod tldr;

/// Bumped whenever parsing behaviour changes.
///
/// The index stores this alongside each row's `source_hash`. A parser fix must invalidate
/// previously harvested rows even though every source file is byte-identical, otherwise
/// the index silently keeps serving output built by older, buggier code.
pub const PARSER_VERSION: u32 = 2;

/// Rendering width. Wide enough that descriptions rarely wrap at all.
const MAN_WIDTH: &str = "400";

/// Sections that mention flags without defining them.
///
/// `ls` and many GNU tools define their options under DESCRIPTION rather than OPTIONS, so
/// this is a denylist rather than an allowlist. An allowlist would silently lose every
/// `ls` flag.
const SKIP_SECTIONS: &[&str] = &[
    "EXAMPLES",
    "EXAMPLE",
    "SEE ALSO",
    "AUTHOR",
    "AUTHORS",
    "COPYRIGHT",
    "REPORTING BUGS",
    "BUGS",
    "HISTORY",
    "NOTES",
    "CAVEATS",
    "FILES",
    "EXIT STATUS",
    "RETURN VALUE",
    "STANDARDS",
];

/// One flag as written in the page. `short` and `long` are two spellings of one flag, so
/// `-r, --recursive` is a single record rather than two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFlag {
    /// Exact short spelling including the leading dash, case preserved: `-r` and `-R` are
    /// different flags and must never be folded together.
    pub short: Option<String>,
    pub long: Option<String>,
    /// Placeholder name for the flag's argument, e.g. `SIZE` in `--block-size=SIZE`.
    pub arg_type: Option<String>,
    /// False when the page writes the argument as optional, e.g. `--color[=WHEN]`.
    pub arg_required: bool,
    pub description: String,
    /// Subsection the flag was defined under, preserved for grouped display.
    pub group: Option<String>,
    /// 1-based line in the rendered page, used for citations like `grep(1):142`.
    pub source_line: usize,
    /// The tag and body lines exactly as rendered, for `shelliq source` to quote verbatim.
    /// Unlike `description`, this is not reflowed onto one line.
    pub excerpt: String,
}

impl ParsedFlag {
    /// How the flag is displayed: `-r, --recursive`.
    pub fn spelling(&self) -> String {
        match (&self.short, &self.long) {
            (Some(s), Some(l)) => format!("{s}, {l}"),
            (Some(s), None) => s.clone(),
            (None, Some(l)) => l.clone(),
            (None, None) => String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedCommand {
    pub name: String,
    pub section: String,
    pub platform: String,
    pub synopsis: String,
    pub description: String,
    pub source_path: String,
    pub source_hash: String,
    pub flags: Vec<ParsedFlag>,
}

pub fn platform() -> &'static str {
    if cfg!(target_os = "macos") { "darwin" } else { "linux" }
}

/// Which kind of executable a name resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecKind {
    /// A regular file on `PATH`.
    File,
    /// A shell builtin, which shadows any same-named file on `PATH`.
    Builtin,
    /// Neither: the name would not run.
    Absent,
}

impl ExecKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecKind::File => "file",
            ExecKind::Builtin => "builtin",
            ExecKind::Absent => "absent",
        }
    }
}

impl std::str::FromStr for ExecKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "file" => Ok(ExecKind::File),
            "builtin" => Ok(ExecKind::Builtin),
            "absent" => Ok(ExecKind::Absent),
            other => bail!("unknown exec_kind `{other}`"),
        }
    }
}

/// Common builtins across bash, zsh, and POSIX sh.
///
/// Not exhaustive — a shell's real builtin set also depends on aliases and functions this
/// process cannot see — but a name on this list is a builtin in every shell shelliq targets,
/// and shadows a same-named file on `PATH` the way a real shell would.
const SHELL_BUILTINS: &[&str] = &[
    "cd", "echo", "export", "unset", "alias", "unalias", "source", ".", "eval", "exec", "exit", "pwd", "read", "set", "shift",
    "test", "[", "true", "false", "type", "history", "jobs", "kill", "wait", "trap", "umask", "ulimit", "let", "local",
    "declare", "typeset", "readonly", "return", "break", "continue", "printf", "getopts", "hash", "bg", "fg", "disown",
    "suspend", "times", "command", "builtin", "enable", "help",
];

/// Executable command names visible through the current `PATH`, plus the builtins ShellIQ
/// knows how to resolve.
///
/// Discovery only reads directory entries and metadata. It never starts a discovered
/// executable, which makes it safe to use as the first stage of system indexing.
pub fn discover_path_commands() -> Vec<String> {
    discover_commands_in_path(std::env::var_os("PATH").as_deref())
}

/// Path-argument form of [`discover_path_commands`], primarily for deterministic callers
/// and tests that must not mutate the process-global `PATH`.
pub fn discover_commands_in_path(path: Option<&std::ffi::OsStr>) -> Vec<String> {
    let mut names: BTreeSet<String> = SHELL_BUILTINS.iter().map(|name| (*name).to_owned()).collect();
    let Some(path) = path else {
        return names.into_iter().collect();
    };

    for directory in std::env::split_paths(path) {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_file() && is_executable(&metadata) {
                names.insert(name);
            }
        }
    }
    names.into_iter().collect()
}

/// The executable identity a shell would run for this name, right now.
///
/// This is deliberately narrower than "does a man page exist for this name": a target is
/// the file (or builtin) that would actually execute, resolved by `PATH` precedence exactly
/// as a shell resolves it, so facts harvested for one installed `grep` are never served for
/// a different `grep` found on a different machine's `PATH`.
///
/// `hash` controls whether the (relatively expensive) content hash is computed. Callers
/// doing a one-off staleness check at lookup time pass `false` and compare size/mtime only;
/// callers doing an explicit harvest or refresh pass `true` to record a verifiable identity.
pub fn resolve_target(name: &str, hash: bool) -> Target {
    if SHELL_BUILTINS.contains(&name) {
        return Target {
            name: name.to_string(),
            platform: platform().to_string(),
            exec_kind: ExecKind::Builtin,
            exec_path: None,
            exec_hash: None,
            exec_size: None,
            exec_mtime: None,
        };
    }

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(name);
            let Ok(meta) = std::fs::metadata(&candidate) else {
                continue;
            };
            if !meta.is_file() || !is_executable(&meta) {
                continue;
            }
            let exec_hash = if hash {
                std::fs::read(&candidate).ok().map(|bytes| hex(&Sha256::digest(&bytes)))
            } else {
                None
            };
            return Target {
                name: name.to_string(),
                platform: platform().to_string(),
                exec_kind: ExecKind::File,
                exec_path: Some(candidate.to_string_lossy().into_owned()),
                exec_hash,
                exec_size: Some(meta.len()),
                exec_mtime: mtime_secs(&meta),
            };
        }
    }

    Target {
        name: name.to_string(),
        platform: platform().to_string(),
        exec_kind: ExecKind::Absent,
        exec_path: None,
        exec_hash: None,
        exec_size: None,
        exec_mtime: None,
    }
}

fn is_executable(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o111 != 0
}

fn mtime_secs(meta: &std::fs::Metadata) -> Option<i64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// The resolved identity of an executable, as of the moment it was resolved.
#[derive(Debug, Clone)]
pub struct Target {
    pub name: String,
    pub platform: String,
    pub exec_kind: ExecKind,
    /// Absolute path, for `ExecKind::File`; `None` otherwise.
    pub exec_path: Option<String>,
    /// sha256 of the file's contents. `None` unless explicitly requested.
    pub exec_hash: Option<String>,
    pub exec_size: Option<u64>,
    pub exec_mtime: Option<i64>,
}

/// Remove control characters a rendered man page should never contain.
///
/// A malformed or hostile page could embed them, and every harvested string ends up on a
/// terminal or in a citation, so they are stripped at ingest rather than trusted downstream.
/// Tab and newline are kept: they are structural, not an injection vector.
pub fn strip_control_chars(s: &str) -> String {
    s.chars().filter(|c| !c.is_control() || *c == '\t' || *c == '\n').collect()
}

/// Preference order when one name has pages in several sections.
///
/// A CLI assistant wants the *command*, so user commands (1) and admin commands (8) come
/// first, then games (6), then the overview and file-format pages, and only then the
/// syscall and library sections. Without this, `kill` resolves to the section 2 syscall
/// rather than the shell command, and `mount` to section 8 by accident rather than by
/// intent.
const SECTION_PREFERENCE: &[&str] = &["1", "8", "6", "7", "5", "2", "3"];

/// Rank a section for command lookup. Lower sorts first.
pub fn section_rank(section: &str) -> usize {
    SECTION_PREFERENCE
        .iter()
        .position(|s| *s == section)
        .unwrap_or(SECTION_PREFERENCE.len())
}

/// Path to the default man page source file, via `man -w`.
pub fn man_path(name: &str) -> Result<String> {
    man_paths(name)?
        .into_iter()
        .next()
        .context(format!("no man page for `{name}`"))
}

/// Every man page source file for this name, across all sections.
///
/// `man -w` alone returns only the default section, which silently discards content: on
/// this machine `signal` resolves to the 84-line section 2 syscall page while the
/// 378-line section 7 overview goes unindexed. `time` has pages in sections 1, 2, and 7.
pub fn man_paths(name: &str) -> Result<Vec<String>> {
    let out = Command::new("man")
        .args(["-w", "-a", name])
        .output()
        .context("running `man -w -a`")?;
    if !out.status.success() {
        bail!("no man page for `{name}`");
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut paths: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if paths.is_empty() {
        bail!("no man page for `{name}`");
    }
    paths.sort_by_key(|p| section_rank(&section_from_path(p)));
    Ok(paths)
}

/// Render a man page to plain text.
///
/// Hyphenation and justification are disabled so that words are never split across lines
/// and runs of spaces inside a description are not padding artifacts — the parser relies
/// on a run of two or more spaces meaning "the tag ended here".
pub fn render(name: &str) -> Result<String> {
    render_section(name, None)
}

/// Render a specific section, or the default one when `section` is `None`.
pub fn render_section(name: &str, section: Option<&str>) -> Result<String> {
    let mut cmd = Command::new("man");
    cmd.env("MANWIDTH", MAN_WIDTH)
        .env("LC_ALL", "C")
        .env("MANPAGER", "cat")
        .env("PAGER", "cat");
    // GNU man-db provides deterministic layout controls that Apple's BSD man
    // does not recognize. macOS still emits parseable output through `cat`,
    // and `strip_overstrike` below removes its terminal formatting.
    #[cfg(target_os = "linux")]
    cmd.args(["--no-hyphenation", "--no-justification", "--pager", "cat"]);
    if let Some(s) = section {
        cmd.arg(s);
    }
    cmd.arg(name);

    let out = cmd.output().context("running `man`")?;
    if !out.status.success() {
        bail!("no man page for `{name}`");
    }
    Ok(strip_overstrike(&String::from_utf8_lossy(&out.stdout)))
}

/// Remove `X\bX` bold and `_\bX` underline sequences.
///
/// GNU man strips these when output is not a terminal, but other implementations do not,
/// and macOS `man` is one of them.
fn strip_overstrike(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i + 1] == '\u{8}' {
            // Keep the character overstruck on top, not the one underneath.
            out.push(chars[i + 2]);
            i += 3;
        } else if chars[i] == '\u{8}' {
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Harvest the preferred man page for a name — the command, not the syscall.
pub fn harvest(name: &str) -> Result<ParsedCommand> {
    let path = man_path(name)?;
    harvest_path(name, &path)
}

/// Harvest every section that documents this name.
///
/// Returned in `SECTION_PREFERENCE` order, so the first entry is the one a shell user
/// most likely means. Sections that fail to render are skipped rather than aborting the
/// whole name.
pub fn harvest_all(name: &str) -> Result<Vec<ParsedCommand>> {
    let paths = man_paths(name)?;
    let harvested: Vec<ParsedCommand> = paths.iter().filter_map(|p| harvest_path(name, p).ok()).collect();
    if harvested.is_empty() {
        bail!("no man page for `{name}` could be parsed");
    }
    Ok(harvested)
}

fn harvest_path(name: &str, path: &str) -> Result<ParsedCommand> {
    let section = section_from_path(path);
    let rendered = render_section(name, Some(&section))?;
    let bytes = std::fs::read(path).unwrap_or_default();
    let source_hash = hex(&Sha256::digest(&bytes));
    Ok(parse_rendered(name, &section, path, &source_hash, &rendered))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// `/usr/share/man/man1/ls.1.gz` -> `1`
fn section_from_path(path: &str) -> String {
    path.rsplit('/')
        .next()
        .map(|f| f.strip_suffix(".gz").unwrap_or(f))
        .and_then(|f| f.rsplit('.').next())
        .filter(|s| !s.is_empty() && s.starts_with(|c: char| c.is_ascii_digit()))
        .unwrap_or("1")
        .to_string()
}

/// Parse rendered man text into a command record.
pub fn parse_rendered(name: &str, section: &str, source_path: &str, source_hash: &str, text: &str) -> ParsedCommand {
    let lines: Vec<&str> = text.lines().collect();
    let mut flags: Vec<ParsedFlag> = Vec::new();
    let mut synopsis = String::new();
    let mut description = String::new();

    let mut section_name = String::new();
    let mut group: Option<String> = None;
    let mut tag_indent: Option<usize> = None;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let indent = indent_of(line);
        let trimmed = line.trim_end();

        // Section heading: flush left.
        if indent == 0 && !trimmed.is_empty() {
            section_name = trimmed.trim().to_string();
            group = None;
            tag_indent = option_tag_indent(&lines, i + 1);
            i += 1;
            continue;
        }

        // Subsection heading sits between the section margin and the tag margin.
        if indent > 0 && tag_indent.is_some_and(|tag_indent| indent < tag_indent) && !trimmed.is_empty() {
            group = Some(trimmed.trim().to_string());
            i += 1;
            continue;
        }

        if section_name == "SYNOPSIS" && synopsis.is_empty() && indent > 0 {
            synopsis = trimmed.trim().to_string();
        }
        if section_name == "DESCRIPTION" && description.is_empty() && indent > 0 {
            let t = trimmed.trim();
            if !t.starts_with('-') {
                description = t.to_string();
            }
        }

        let skipped = SKIP_SECTIONS.contains(&section_name.as_str());
        if !skipped && tag_indent == Some(indent) && looks_like_option_tag(&lines, i) {
            let content = trimmed.trim_start();
            let (spec, inline) = split_tag(content);
            if let Some(mut parsed) = parse_spec(spec) {
                let mut alias_specs = vec![spec];
                let mut desc_parts: Vec<String> = Vec::new();
                let mut excerpt_lines: Vec<&str> = vec![line];
                if !inline.trim().is_empty() {
                    if inline.trim_start().starts_with('-') {
                        alias_specs.push(inline.trim());
                    } else {
                        desc_parts.push(inline.trim().to_string());
                    }
                }
                // Body lines sit at BODY_INDENT until a blank line ends the entry.
                let mut j = i + 1;
                while j < lines.len() {
                    let b = lines[j];
                    if b.trim().is_empty() || indent_of(b) <= indent {
                        break;
                    }
                    excerpt_lines.push(b);
                    if desc_parts.is_empty() && b.trim_start().starts_with('-') {
                        alias_specs.push(b.trim());
                    } else {
                        desc_parts.push(b.trim().to_string());
                    }
                    j += 1;
                }
                parsed.description = strip_control_chars(&desc_parts.join(" "));
                parsed.group = group.clone().map(|g| strip_control_chars(&g));
                parsed.source_line = i + 1;
                parsed.excerpt = strip_control_chars(&excerpt_lines.join("\n"));
                let aliases: Vec<_> = alias_specs
                    .into_iter()
                    .flat_map(|alias_spec| additional_aliases(alias_spec, &parsed))
                    .collect();
                push_unique(&mut flags, parsed);
                for alias in aliases {
                    push_unique(&mut flags, alias);
                }
                i = j;
                continue;
            }
        }

        i += 1;
    }

    ParsedCommand {
        name: name.to_string(),
        section: section.to_string(),
        platform: platform().to_string(),
        synopsis: strip_control_chars(&synopsis),
        description: strip_control_chars(&description),
        source_path: source_path.to_string(),
        source_hash: source_hash.to_string(),
        flags,
    }
}

/// Merge a flag into the list, combining spellings that describe the same option.
///
/// A page may document `-r` and `--recursive` on separate lines; keeping both as one
/// record means a lookup for either spelling finds the same facts.
pub(crate) fn push_unique(flags: &mut Vec<ParsedFlag>, incoming: ParsedFlag) {
    let clash = flags
        .iter_mut()
        .find(|f| (f.short.is_some() && f.short == incoming.short) || (f.long.is_some() && f.long == incoming.long));
    match clash {
        Some(existing) => {
            if existing.short.is_none() {
                existing.short = incoming.short;
            }
            if existing.long.is_none() {
                existing.long = incoming.long;
            }
            if existing.description.is_empty() {
                existing.description = incoming.description;
            }
            if existing.excerpt.is_empty() {
                existing.excerpt = incoming.excerpt;
            }
        }
        None => flags.push(incoming),
    }
}

fn indent_of(line: &str) -> usize {
    line.chars()
        .take_while(|character| matches!(character, ' ' | '\t'))
        .fold(
            0,
            |column, character| {
                if character == '\t' { (column / 8 + 1) * 8 } else { column + 1 }
            },
        )
}

/// Discover the option-tag margin used by the current rendered section.
///
/// GNU man-db and BSD mandoc choose different absolute columns. In both
/// renderings, continuation and description lines are deeper than the tag, so
/// the shallowest flag-shaped line is the stable structural boundary.
fn option_tag_indent(lines: &[&str], start: usize) -> Option<usize> {
    lines[start..]
        .iter()
        .enumerate()
        .take_while(|(_, line)| indent_of(line) > 0 || line.trim().is_empty())
        .filter(|(offset, _)| looks_like_option_tag(lines, start + offset))
        .map(|(_, line)| indent_of(line))
        .min()
}

/// A definition tag introduces a more deeply indented body, or carries its
/// description after spacing on the same line. This excludes command examples
/// such as curl's standalone `--expand-url = ...` line.
fn looks_like_option_tag(lines: &[&str], index: usize) -> bool {
    let Some(line) = lines.get(index) else {
        return false;
    };
    let content = line.trim_start();
    if !content.starts_with('-') {
        return false;
    }
    let (spec, inline) = split_tag(content);
    if parse_spec(spec).is_none() {
        return false;
    }
    if !inline.trim().is_empty() && !inline.trim_start().starts_with('-') {
        return true;
    }
    if has_unpadded_inline_description(content) {
        return true;
    }
    lines[index + 1..]
        .iter()
        .find(|body| !body.trim().is_empty())
        .is_some_and(|body| indent_of(body) > indent_of(line))
}

/// BSD mandoc and a few GNU pages use one space between a flag and a short
/// inline description. Argument placeholders and shell assignment examples are
/// deliberately excluded.
fn has_unpadded_inline_description(content: &str) -> bool {
    let Some((_, tail)) = content.split_once(char::is_whitespace) else {
        return false;
    };
    let tail = tail.trim_start();
    if tail.is_empty()
        || !tail.contains(char::is_whitespace)
        || tail.starts_with(['-', '=', '<', '[', '\'', '"'])
        || tail
            .chars()
            .all(|character| character.is_ascii_uppercase() || character.is_ascii_digit() || matches!(character, '_' | '-' | '.'))
    {
        return false;
    }
    true
}

/// Split a `.TP` tag line into its flag spec and any description on the same line.
///
/// A run of two or more spaces separates them. Justification is disabled during rendering
/// precisely so that such a run is never an artifact of padding.
pub(crate) fn split_tag(content: &str) -> (&str, &str) {
    match content.find("  ") {
        Some(i) => (&content[..i], &content[i..]),
        None => (content, ""),
    }
}

/// Parse `-A NUM, --after-context=NUM` into one flag record.
pub(crate) fn parse_spec(spec: &str) -> Option<ParsedFlag> {
    let spec = spec.trim();
    if !spec.starts_with('-') {
        return None;
    }

    let mut flag = ParsedFlag {
        short: None,
        long: None,
        arg_type: None,
        arg_required: true,
        description: String::new(),
        group: None,
        source_line: 0,
        excerpt: String::new(),
    };
    let mut saw_any = false;

    for part in spec.split(',') {
        let part = part.trim();
        if !part.starts_with('-') || part == "-" || part == "--" {
            continue;
        }
        let (name, arg, required) = split_arg(part);
        if name.len() < 2 || !is_flag_name(name) {
            continue;
        }
        if name.starts_with("--") {
            if flag.long.is_none() {
                flag.long = Some(name.to_string());
            }
        } else if flag.short.is_none() {
            flag.short = Some(name.to_string());
        }
        if let Some(a) = arg
            && flag.arg_type.is_none()
        {
            flag.arg_type = Some(a.to_string());
            flag.arg_required = required;
        }
        saw_any = true;
    }

    saw_any.then_some(flag)
}

/// Return additional spellings from a tag that cannot fit in the primary
/// one-short/one-long record, preserving the same facts and citation.
fn additional_aliases(spec: &str, template: &ParsedFlag) -> Vec<ParsedFlag> {
    spec.split(',')
        .flat_map(|part| part.split_whitespace().take_while(|token| token.starts_with('-')))
        .filter_map(|part| {
            let part = part.trim().trim_end_matches(',');
            if !part.starts_with('-') || part == "-" || part == "--" {
                return None;
            }
            let (name, _, _) = split_arg(part);
            if name.len() < 2 || !is_flag_name(name) {
                return None;
            }
            if template.short.as_deref() == Some(name) || template.long.as_deref() == Some(name) {
                return None;
            }

            let mut alias = template.clone();
            alias.short = None;
            alias.long = None;
            if name.starts_with("--") {
                alias.long = Some(name.to_string());
            } else {
                alias.short = Some(name.to_string());
            }
            Some(alias)
        })
        .collect()
}

/// Separate a flag from its argument: `--color[=WHEN]`, `--width=COLS`, `-w COLS`,
/// `--form <name=content>`.
///
/// Returns the flag name, the argument placeholder, and whether it is required.
///
/// The delimiters are checked in positional order rather than a fixed priority. Splitting
/// on `=` first would break `--form <name=content>`, whose `=` sits inside the placeholder
/// and belongs to the argument, not to the flag.
fn split_arg(part: &str) -> (&str, Option<&str>, bool) {
    let bracket = part.find("[=");
    let eq = part.find('=');
    let space = part.find(' ');
    let first = [bracket, eq, space].into_iter().flatten().min();

    match first {
        Some(i) if Some(i) == bracket => {
            let arg = part[i + 2..].trim_end_matches(']');
            (&part[..i], Some(arg), false)
        }
        Some(i) if Some(i) == space => {
            let raw = part[i + 1..].trim();
            let optional = raw.starts_with('[');
            let arg = raw.trim_matches(|c| c == '[' || c == ']' || c == '<' || c == '>');
            if arg.is_empty() {
                (&part[..i], None, true)
            } else if raw.contains(char::is_whitespace) && !raw.starts_with(['[', '<']) && !raw.ends_with(';') {
                // Rendered man pages occasionally put prose on the tag line (`-print
                // True; print ...`). Unwrapped multi-word text is a description, not an
                // argument placeholder. Preserve command-list placeholders such as
                // `-exec command ;` and explicitly delimited forms.
                (&part[..i], None, true)
            } else {
                (&part[..i], Some(arg), !optional)
            }
        }
        Some(i) => (&part[..i], Some(&part[i + 1..]), true),
        None => (part, None, true),
    }
}

/// Reject prose that merely begins with a dash, e.g. `-- and then some text`.
///
/// A flag name is dashes followed by alphanumerics, `-`, `_`, or `.`, and must contain at
/// least one alphanumeric character. `grep`'s numeric `-NUM` form satisfies this, and `.`
/// is allowed for version-bearing names like `curl --http1.1` and `--tlsv1.2`.
fn is_flag_name(name: &str) -> bool {
    let body = name.trim_start_matches('-');
    !body.is_empty()
        && body.chars().any(|c| c.is_ascii_alphanumeric())
        && body
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn path_discovery_includes_only_executable_files_and_deduplicates_names() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "shelliq-path-discovery-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let first = root.join("first");
        let second = root.join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        for path in [first.join("alpha"), second.join("alpha"), second.join("beta")] {
            std::fs::write(&path, b"").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        std::fs::write(first.join("not-executable"), b"").unwrap();
        std::fs::create_dir(first.join("directory")).unwrap();

        let joined = std::env::join_paths([&first, &second]).unwrap();
        let names = discover_commands_in_path(Some(&joined));
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(names.iter().filter(|name| name.as_str() == "alpha").count(), 1);
        assert!(names.iter().any(|name| name == "beta"));
        assert!(!names.iter().any(|name| name == "not-executable"));
        assert!(!names.iter().any(|name| name == "directory"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn splits_flag_from_inline_description() {
        let (spec, desc) = split_tag("-c     with -lt: sort by ctime");
        assert_eq!(spec, "-c");
        assert_eq!(desc.trim(), "with -lt: sort by ctime");
    }

    #[test]
    fn tag_with_no_inline_description() {
        let (spec, desc) = split_tag("-b, --escape");
        assert_eq!(spec, "-b, --escape");
        assert_eq!(desc, "");
    }

    #[test]
    fn parses_short_and_long_together() {
        let f = parse_spec("-b, --escape").unwrap();
        assert_eq!(f.short.as_deref(), Some("-b"));
        assert_eq!(f.long.as_deref(), Some("--escape"));
        assert_eq!(f.arg_type, None);
    }

    #[test]
    fn parses_required_argument() {
        let f = parse_spec("--block-size=SIZE").unwrap();
        assert_eq!(f.long.as_deref(), Some("--block-size"));
        assert_eq!(f.arg_type.as_deref(), Some("SIZE"));
        assert!(f.arg_required);
    }

    #[test]
    fn parses_optional_argument() {
        let f = parse_spec("--color[=WHEN]").unwrap();
        assert_eq!(f.long.as_deref(), Some("--color"));
        assert_eq!(f.arg_type.as_deref(), Some("WHEN"));
        assert!(!f.arg_required);
    }

    #[test]
    fn parses_separated_argument() {
        let f = parse_spec("-A NUM, --after-context=NUM").unwrap();
        assert_eq!(f.short.as_deref(), Some("-A"));
        assert_eq!(f.long.as_deref(), Some("--after-context"));
        assert_eq!(f.arg_type.as_deref(), Some("NUM"));
    }

    #[test]
    fn prose_after_a_single_dash_option_is_not_an_argument_placeholder() {
        let flag = parse_spec("-print True; print the full file name").unwrap();
        assert_eq!(flag.short.as_deref(), Some("-print"));
        assert_eq!(flag.arg_type, None);
    }

    #[test]
    fn command_list_placeholder_with_spaces_is_preserved() {
        let flag = parse_spec("-exec command ;").unwrap();
        assert_eq!(flag.arg_type.as_deref(), Some("command ;"));
    }

    #[test]
    fn case_is_preserved_as_distinct_flags() {
        let lower = parse_spec("-r, --recursive").unwrap();
        let upper = parse_spec("-R, --dereference-recursive").unwrap();
        assert_ne!(lower.short, upper.short);
    }

    #[test]
    fn grouped_short_aliases_are_all_preserved() {
        let parsed = parse_rendered(
            "grep",
            "1",
            "/usr/share/man/man1/grep.1",
            "hash",
            "OPTIONS\n       -R,\n              -r, --recursive\n              Search directories recursively.\n",
        );
        let lower = parsed.flags.iter().find(|flag| flag.short.as_deref() == Some("-r"));
        let upper = parsed.flags.iter().find(|flag| flag.short.as_deref() == Some("-R"));
        assert!(lower.is_some(), "grouped -r alias must be indexed");
        assert!(upper.is_some(), "grouped -R alias must be indexed");
        assert_eq!(lower.unwrap().source_line, upper.unwrap().source_line);
        assert_eq!(lower.unwrap().description, "Search directories recursively.");
    }

    #[test]
    fn discovers_gnu_and_macos_option_margins() {
        for fixture in [
            "OPTIONS\n       -R, -r, --recursive\n              Search directories recursively.\n",
            "OPTIONS\n     -R, -r, --recursive\n             Search directories recursively.\n",
        ] {
            let parsed = parse_rendered("grep", "1", "grep.1", "hash", fixture);
            assert!(parsed.flags.iter().any(|flag| flag.short.as_deref() == Some("-R")));
            assert!(parsed.flags.iter().any(|flag| flag.short.as_deref() == Some("-r")));
            assert!(parsed.flags.iter().any(|flag| flag.long.as_deref() == Some("--recursive")));
        }
    }

    #[test]
    fn preserves_aliases_when_renderer_omits_commas() {
        let parsed = parse_rendered(
            "grep",
            "1",
            "grep.1",
            "hash",
            "OPTIONS\n     -R -r --recursive\n             Search directories recursively.\n",
        );
        assert!(parsed.flags.iter().any(|flag| flag.short.as_deref() == Some("-R")));
        assert!(parsed.flags.iter().any(|flag| flag.short.as_deref() == Some("-r")));
        assert!(parsed.flags.iter().any(|flag| flag.long.as_deref() == Some("--recursive")));
    }

    #[test]
    fn tab_indented_command_examples_are_not_option_tags() {
        let parsed = parse_rendered(
            "curl",
            "1",
            "curl.1",
            "hash",
            "DESCRIPTION\n       --variable <name=content>\n              Set a variable.\n\n\t --expand-variable fix@{{HOME}}/.secret\n\t https://example.com/\n",
        );
        assert!(parsed.flags.iter().any(|flag| flag.long.as_deref() == Some("--variable")));
        assert!(
            !parsed
                .flags
                .iter()
                .any(|flag| flag.long.as_deref() == Some("--expand-variable"))
        );
    }

    #[test]
    fn rejects_prose_beginning_with_a_dash() {
        assert!(parse_spec("- this is prose, not a flag").is_none());
        assert!(parse_spec("-- and then some text").is_none());
    }

    #[test]
    fn strips_overstrike_bold() {
        assert_eq!(strip_overstrike("a\u{8}ab\u{8}b"), "ab");
    }

    #[test]
    fn section_parsed_from_path() {
        assert_eq!(section_from_path("/usr/share/man/man1/ls.1.gz"), "1");
        assert_eq!(section_from_path("/usr/share/man/man5/passwd.5.gz"), "5");
    }
}
