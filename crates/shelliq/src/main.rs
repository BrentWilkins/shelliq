//! shelliq — a local, verifiable CLI assistant.
//!
//! Tier 0: everything here answers from the SQLite index alone. No model, no network, and
//! no inference library is linked into this binary.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use shelliq_index::Index;
use shelliq_verify::Finding;

#[derive(Parser)]
#[command(
    name = "shelliq",
    version,
    about = "Answers about command flags, from the man pages on this machine"
)]
struct Cli {
    /// Index file. Defaults to $SHELLIQ_INDEX or ~/.local/share/shelliq/index.sqlite
    #[arg(long, global = true)]
    index: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Harvest man pages into the index.
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },
    /// Explain a command line, checking every flag against the index.
    Explain {
        /// The command line, e.g. `grep -r` or `tar -xzf archive.tar.gz`
        #[arg(required = true, trailing_var_arg = true)]
        line: Vec<String>,
    },
    /// Find a flag by what it does: `shelliq search curl follow redirect`
    Search {
        command: String,
        #[arg(required = true, trailing_var_arg = true)]
        query: Vec<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// List every flag for a command.
    Flags { command: String },
    /// Show exactly where a citation such as `grep(1):168` came from.
    Source { citation: String },
}

#[derive(Subcommand)]
enum IndexAction {
    /// Harvest the named commands. Every man section for a name is indexed.
    Build {
        #[arg(required = true)]
        names: Vec<String>,
    },
    /// Re-harvest every indexed name (or just the ones given) into a fresh index, then
    /// atomically replace the old one.
    Refresh { names: Vec<String> },
    /// Show index size.
    Stats,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = cli.index.unwrap_or_else(Index::default_path);

    match cli.command {
        Command::Index { action } => match action {
            IndexAction::Build { names } => build(&path, &names),
            IndexAction::Refresh { names } => refresh(&path, &names),
            IndexAction::Stats => stats(&path),
        },
        Command::Explain { line } => explain(&path, &line.join(" ")),
        Command::Search { command, query, limit } => search(&path, &command, &query.join(" "), limit),
        Command::Flags { command } => list_flags(&path, &command),
        Command::Source { citation } => source(&path, &citation),
    }
}

fn build(path: &std::path::Path, names: &[String]) -> Result<()> {
    let mut index = Index::open(path)?;
    let mut pages = 0usize;
    let mut flags = 0usize;

    for name in names {
        let target = shelliq_harvest::resolve_target(name, true);
        match shelliq_harvest::harvest_all(name) {
            Ok(commands) => {
                for cmd in commands {
                    flags += cmd.flags.len();
                    pages += 1;
                    println!("  {}({})  {} flags", cmd.name, cmd.section, cmd.flags.len());
                    index.insert_command(&target, &cmd)?;
                }
            }
            Err(e) => eprintln!("  {name}: {e}"),
        }
    }

    println!("\nindexed {pages} pages, {flags} flags -> {}", path.display());
    shelliq_index::secure_permissions(path);
    Ok(())
}

/// Rebuild every harvested name into a sibling file, then swap it in atomically.
///
/// A refresh in place would leave a reader briefly looking at a half-rewritten index; this
/// instead only ever replaces the whole file in one `rename`, and a name that no longer
/// resolves to anything simply harvests nothing and drops out of the fresh index, so a
/// removed command's stale facts do not linger.
fn refresh(path: &std::path::Path, names: &[String]) -> Result<()> {
    let names: Vec<String> = if names.is_empty() {
        Index::open(path)?.all_target_names()?
    } else {
        names.to_vec()
    };
    if names.is_empty() {
        println!("nothing to refresh");
        return Ok(());
    }

    let tmp_path = path.with_extension("sqlite.refresh");
    let _ = std::fs::remove_file(&tmp_path);
    let _ = std::fs::remove_file(format!("{}-wal", tmp_path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", tmp_path.display()));

    let mut fresh = Index::open(&tmp_path)?;
    let mut pages = 0usize;
    for name in &names {
        let target = shelliq_harvest::resolve_target(name, true);
        match shelliq_harvest::harvest_all(name) {
            Ok(commands) => {
                for cmd in commands {
                    pages += 1;
                    fresh.insert_command(&target, &cmd)?;
                }
            }
            Err(e) => eprintln!("  {name}: {e}"),
        }
    }
    fresh.checkpoint_and_close()?;
    drop(fresh);
    shelliq_index::secure_permissions(&tmp_path);

    std::fs::rename(&tmp_path, path).context("swapping in the refreshed index")?;
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }

    println!(
        "refreshed {} pages across {} name(s) -> {}",
        pages,
        names.len(),
        path.display()
    );
    Ok(())
}

fn source(path: &std::path::Path, citation: &str) -> Result<()> {
    let (command, section, line) = parse_citation(citation)?;
    let index = Index::open(path)?;
    let Some(prov) = index.provenance(&command, &section, line)? else {
        anyhow::bail!("no citation `{citation}` in the index");
    };

    println!("source:    man page");
    println!("path:      {}", prov.source_path);
    println!("hash:      sha256:{}", prov.source_hash);
    println!("exec:      {} {}", prov.exec_kind, prov.exec_path.as_deref().unwrap_or("-"));
    if let Some(hash) = &prov.exec_hash {
        println!("exec hash: sha256:{hash}");
    }
    println!("anchor:    {}({}):{}", prov.command, prov.section, prov.source_line);
    println!();
    println!("{}", prov.excerpt);
    Ok(())
}

/// Print a caveat when the live executable no longer matches what was harvested.
///
/// A stale note is not an abstention: the facts are still shown, since they are usually
/// still right, but the caller is told exactly what to run when they are not.
fn warn_if_stale(index: &Index, command: &str) -> Result<()> {
    if index.freshness(command)? == shelliq_index::Freshness::PossiblyStale {
        println!(
            "note: `{command}` on PATH looks different from when it was indexed; \
             run `shelliq index refresh {command}`\n"
        );
    }
    Ok(())
}

/// Parse `grep(1):168` into its command, section, and line.
fn parse_citation(citation: &str) -> Result<(String, String, usize)> {
    let (head, line) = citation
        .rsplit_once(':')
        .with_context(|| format!("`{citation}` is not a citation; expected e.g. grep(1):168"))?;
    let line: usize = line.parse().with_context(|| format!("`{citation}` has a non-numeric line"))?;
    let (command, section) = head
        .strip_suffix(')')
        .and_then(|h| h.rsplit_once('('))
        .with_context(|| format!("`{citation}` is not a citation; expected e.g. grep(1):168"))?;
    Ok((command.to_string(), section.to_string(), line))
}

fn stats(path: &std::path::Path) -> Result<()> {
    let index = Index::open(path)?;
    let (commands, flags) = index.stats()?;
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("{commands} pages, {flags} flags, {} KiB", size / 1024);
    println!("{}", path.display());
    Ok(())
}

fn explain(path: &std::path::Path, line: &str) -> Result<()> {
    let index = Index::open(path)?;

    if let Ok(segments) = shelliq_verify::segments(line) {
        let mut warned = std::collections::BTreeSet::new();
        for segment in &segments {
            if !segment.command.opaque && warned.insert(segment.command.text.clone()) {
                warn_if_stale(&index, &segment.command.text)?;
            }
        }
    }

    let findings = shelliq_verify::verify(&index, line).with_context(|| format!("verifying `{line}`"))?;

    if findings.is_empty() {
        println!("nothing to check in `{line}`");
        return Ok(());
    }

    let mut problems = 0;
    for finding in &findings {
        if !finding.is_clean() {
            problems += 1;
        }
        println!("{}", render::finding(finding));
    }

    if problems > 0 {
        println!("\n{problems} of {} not confirmed", findings.len());
        std::process::exit(1);
    }
    Ok(())
}

fn search(path: &std::path::Path, command: &str, query: &str, limit: usize) -> Result<()> {
    let index = Index::open(path)?;
    warn_if_stale(&index, command)?;
    let all = index.flags_for(command)?;
    if all.is_empty() {
        anyhow::bail!("`{command}` is not indexed; run `shelliq index build {command}`");
    }

    let hits = index.search_flags(command, query, limit)?;
    if hits.is_empty() {
        println!("no flag of `{command}` mentions {query:?}");
        return Ok(());
    }

    for hit in &hits {
        println!("{}", render::flag_line(hit));
    }
    // Coverage is stated explicitly: a fuzzy result is never exhaustive.
    println!("\n{} of {}, matched on description", hits.len(), all.len());
    Ok(())
}

fn list_flags(path: &std::path::Path, command: &str) -> Result<()> {
    let index = Index::open(path)?;
    warn_if_stale(&index, command)?;
    let flags = index.flags_for(command)?;
    if flags.is_empty() {
        anyhow::bail!("`{command}` is not indexed; run `shelliq index build {command}`");
    }
    for flag in &flags {
        println!("{}", render::flag_line(flag));
    }
    println!("\n{} flags", flags.len());
    Ok(())
}

/// Terminal output.
///
/// Deliberately plain, line-oriented printing rather than a full-screen TUI. shelliq's
/// output appears inline beneath a shell prompt, next to the zsh line editor, so it must
/// not take over the screen or run an event loop.
mod render {
    use super::Finding;
    use anstyle::{AnsiColor, Style};

    const GREEN: Style = Style::new().fg_color(Some(anstyle::Color::Ansi(AnsiColor::Green)));
    const YELLOW: Style = Style::new().fg_color(Some(anstyle::Color::Ansi(AnsiColor::Yellow)));
    const RED: Style = Style::new().fg_color(Some(anstyle::Color::Ansi(AnsiColor::Red)));
    const DIM: Style = Style::new().dimmed();
    const BOLD: Style = Style::new().bold();

    fn truncate(s: &str, max: usize) -> String {
        if s.chars().count() <= max {
            return s.to_string();
        }
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }

    pub fn flag_line(flag: &shelliq_index::FlagRow) -> String {
        let spelling = match &flag.arg_type {
            Some(a) if flag.arg_required => format!("{} <{a}>", flag.spelling()),
            Some(a) => format!("{} [{a}]", flag.spelling()),
            None => flag.spelling(),
        };
        format!(
            "  {BOLD}{:<28}{BOLD:#} {:<52} {DIM}{}{DIM:#}",
            truncate(&spelling, 28),
            truncate(&flag.description, 52),
            flag.citation()
        )
    }

    pub fn finding(finding: &Finding) -> String {
        match finding {
            Finding::Verified { token, flag } => format!(
                "  {GREEN}✓{GREEN:#} {BOLD}{token}{BOLD:#}  {}  {DIM}{}{DIM:#}",
                truncate(&flag.description, 60),
                flag.citation()
            ),
            Finding::WrongCase { token, suggestion } => format!(
                "  {YELLOW}~{YELLOW:#} {BOLD}{token}{BOLD:#}  not valid for {}; did you mean \
                 {YELLOW}{}{YELLOW:#}?  {DIM}{}{DIM:#}\n      {}",
                suggestion.command,
                suggestion.short.as_deref().unwrap_or_default(),
                suggestion.citation(),
                truncate(&suggestion.description, 70),
            ),
            Finding::UnknownFlag { command, token } => format!(
                "  {RED}✗{RED:#} {BOLD}{token}{BOLD:#}  no such flag for {command} \
                 {DIM}(unverified){DIM:#}"
            ),
            Finding::UnknownCommand { command } => format!(
                "  {RED}?{RED:#} {BOLD}{command}{BOLD:#}  not indexed \
                 {DIM}(run `shelliq index build {command}`){DIM:#}"
            ),
            Finding::MissingArgument { token, flag } => format!(
                "  {YELLOW}!{YELLOW:#} {BOLD}{token}{BOLD:#}  needs an argument <{}>  \
                 {DIM}{}{DIM:#}",
                flag.arg_type.as_deref().unwrap_or("ARG"),
                flag.citation()
            ),
            Finding::Unsupported { construct, text } => format!(
                "  {YELLOW}⊘{YELLOW:#} {BOLD}{}{BOLD:#}  not checked: contains {construct}  \
                 {DIM}(abstained rather than guess){DIM:#}",
                truncate(text, 48),
            ),
        }
    }
}
