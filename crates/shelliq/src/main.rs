//! shelliq — a local, verifiable CLI assistant.
//!
//! Tier 0: everything here answers from the SQLite index alone. No model, no network, and
//! no inference library is linked into this binary.

#[cfg(feature = "model")]
mod documentation;
#[cfg(feature = "model")]
mod model;
#[cfg(feature = "model")]
mod policy;
#[cfg(feature = "model")]
mod response;

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
    Flags {
        command: String,
        /// One flag spelling per line, no colour or citation — for shell completion, not
        /// people. Silent (exit 0, no output) rather than an explanatory error when the
        /// command isn't indexed, since a completer should fall through quietly.
        #[arg(long)]
        raw: bool,
    },
    /// Show exactly where a citation such as `grep(1):168` came from.
    /// Generate an experimental command through a local model server, with a
    /// fail-closed vendored-documentation fallback.
    #[cfg(feature = "model")]
    Suggest {
        /// Natural-language task to turn into an editable command.
        #[arg(required = true, trailing_var_arg = true)]
        instruction: Vec<String>,
        /// OpenAI-compatible loopback chat-completions endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080/v1/chat/completions")]
        endpoint: String,
        /// Retrieved evidence supplied to the model. Empty by default.
        #[arg(long, default_value = "")]
        context: String,
        /// Prompt shape expected by the served adapter.
        #[arg(long, value_enum, default_value_t)]
        prompt_contract: model::PromptContract,
        /// Entire request deadline in milliseconds.
        #[arg(long, default_value_t = 5_000)]
        timeout_ms: u64,
        /// Answer a focused question. Repeat for accumulated continuation answers.
        #[arg(long, value_name = "VALUE")]
        answer: Vec<String>,
        /// Pin answers to the documentation source from a needs_input continuation.
        #[arg(long, value_name = "SOURCE")]
        continue_from: Option<String>,
        /// Print a versioned response envelope for every outcome.
        #[arg(long)]
        json: bool,
        /// Print the stable response protocol consumed by the Zsh widget.
        #[arg(long, conflicts_with = "json")]
        zsh_widget: bool,
    },
    Source {
        citation: String,
    },
}

#[derive(Subcommand)]
enum IndexAction {
    /// Discover executable names on PATH and index their installed man pages.
    /// Discovery never executes the commands it finds.
    Scan,
    /// Harvest the named commands. Every man section for a name is indexed; a name with no
    /// man page falls back to crawling `--help` and its subcommands' `--help`.
    Build {
        #[arg(required = true)]
        names: Vec<String>,
        /// Let the `--help` crawler execute binaries under this directory even though it is
        /// writable by someone other than root, e.g. `~/.cargo/bin`. Repeatable. Empty by
        /// default, so nothing outside a root-owned, locked-down directory runs unopted-in.
        #[arg(long = "allow-writable-path")]
        allow_writable_path: Vec<std::path::PathBuf>,
    },
    /// Re-harvest every indexed name (or just the ones given) into a fresh index, then
    /// atomically replace the old one.
    Refresh {
        names: Vec<String>,
        #[arg(long = "allow-writable-path")]
        allow_writable_path: Vec<std::path::PathBuf>,
    },
    /// Show index size.
    Stats,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = cli.index.unwrap_or_else(Index::default_path);

    match cli.command {
        Command::Index { action } => match action {
            IndexAction::Scan => scan(&path),
            IndexAction::Build {
                names,
                allow_writable_path,
            } => build(&path, &names, &allow_writable_path),
            IndexAction::Refresh {
                names,
                allow_writable_path,
            } => refresh(&path, &names, &allow_writable_path),
            IndexAction::Stats => stats(&path),
        },
        Command::Explain { line } => explain(&path, &line.join(" ")),
        Command::Search { command, query, limit } => search(&path, &command, &query.join(" "), limit),
        Command::Flags { command, raw } => list_flags(&path, &command, raw),
        #[cfg(feature = "model")]
        Command::Suggest {
            instruction,
            endpoint,
            context,
            prompt_contract,
            timeout_ms,
            answer,
            continue_from,
            json,
            zsh_widget,
        } => suggest_command(
            &path,
            &endpoint,
            &context,
            &instruction.join(" "),
            prompt_contract,
            timeout_ms,
            &answer,
            continue_from.as_deref(),
            json,
            zsh_widget,
        ),
        Command::Source { citation } => source(&path, &citation),
    }
}

#[cfg(feature = "model")]
#[allow(clippy::too_many_arguments)]
fn suggest_command(
    path: &std::path::Path,
    endpoint: &str,
    context: &str,
    instruction: &str,
    prompt_contract: model::PromptContract,
    timeout_ms: u64,
    answers: &[String],
    continue_from: Option<&str>,
    json: bool,
    zsh_widget: bool,
) -> Result<()> {
    if timeout_ms == 0 {
        anyhow::bail!("--timeout-ms must be positive");
    }
    let platform = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    if continue_from.is_some() && answers.is_empty() {
        anyhow::bail!("--continue-from requires at least one --answer");
    }
    let mut answered_instruction = instruction.to_owned();
    for answer in answers {
        let answer = answer.trim();
        if answer.is_empty() {
            anyhow::bail!("--answer must not be empty");
        }
        answered_instruction.push_str(". Use ");
        answered_instruction.push_str(answer);
        answered_instruction.push('.');
    }
    let effective_instruction = if answers.is_empty() {
        instruction
    } else {
        &answered_instruction
    };
    let index = Index::open(path)?;
    let shortlist = if continue_from.is_some() {
        Vec::new()
    } else {
        index.search_commands(effective_instruction, 6)?
    };
    if std::env::var_os("SHELLIQ_DEBUG_RETRIEVAL").is_some() {
        eprintln!("retrieved command shortlist: {}", shortlist.join(", "));
    }
    let model_result = if !answers.is_empty() {
        Err(anyhow::anyhow!("clarification answers are recompiled from documentation"))
    } else {
        try_model_suggestion(
            &index,
            endpoint,
            context,
            effective_instruction,
            platform,
            prompt_contract,
            timeout_ms,
            &shortlist,
        )
    };
    let (suggestion, origin, source, documentation_intent) = match model_result {
        Ok(suggestion) => (suggestion, SuggestionOrigin::Model, "model".to_owned(), None),
        Err(model_error) => {
            let fallback = if let Some(source) = continue_from {
                documentation::compile_source(&index, platform, effective_instruction, source)?
            } else {
                documentation::compile(&index, &shortlist, platform, effective_instruction)?
            };
            match fallback.status {
                documentation::CompilationStatus::Ready => {
                    let source = fallback.source.clone().unwrap_or_else(|| "unknown".into());
                    let documentation_intent = fallback.intent.clone();
                    let suggestion = fallback
                        .suggestion
                        .context("ready documentation fallback omitted its command")?;
                    let findings = shelliq_verify::verify(&index, &suggestion.command)?;
                    if findings.iter().any(|finding| !finding.is_clean()) {
                        anyhow::bail!(
                            "model suggestion failed ({model_error:#}); documentation fallback failed local command/flag validation"
                        );
                    }
                    if !answers.is_empty() {
                        eprintln!(
                            "documentation clarification compiled; source {}; selected command {}; command and option spellings locally verified",
                            source,
                            fallback.command_name.as_deref().unwrap_or("unknown")
                        );
                    } else {
                        eprintln!(
                            "documentation fallback after model failure; source {}; selected command {}; command and option spellings locally verified",
                            source,
                            fallback.command_name.as_deref().unwrap_or("unknown")
                        );
                    }
                    (suggestion, SuggestionOrigin::Documentation, source, documentation_intent)
                }
                documentation::CompilationStatus::NeedsInput => {
                    let clarification = fallback
                        .clarification()
                        .context("needs_input response omitted its clarification")?;
                    if zsh_widget {
                        println!(
                            "{}",
                            response::widget_needs_input(&clarification, fallback.source.as_deref())?
                        );
                    } else if json {
                        println!(
                            "{}",
                            response::needs_input(
                                &clarification,
                                fallback.command_name.as_deref(),
                                fallback.source.as_deref(),
                                fallback.intent.as_deref(),
                            )?
                        );
                    } else {
                        eprintln!("needs_input: {}", clarification.question);
                        eprintln!(
                            "Add the answer to your request or pass --continue-from SOURCE --answer VALUE, then try again."
                        );
                    }
                    anyhow::bail!("needs_input");
                }
                documentation::CompilationStatus::NoDocumentation => {
                    let message = "no complete installed documentation recipe matched";
                    if zsh_widget {
                        println!("{}", response::widget_no_documentation(message));
                    } else if json {
                        println!("{}", response::no_documentation(message)?);
                    } else {
                        eprintln!("no_documentation: {message}");
                    }
                    anyhow::bail!("no_documentation: {model_error:#}");
                }
            }
        }
    };

    policy::enforce(
        &index,
        &suggestion.command,
        effective_instruction,
        origin == SuggestionOrigin::Model,
    )
    .context("suggestion rejected by deterministic policy")?;

    if origin == SuggestionOrigin::Model {
        eprintln!(
            "experimental model suggestion; option spellings checked, operand semantics and pipeline compatibility unverified; inspect and edit before running"
        );
    }
    if zsh_widget {
        println!("{}", response::widget_ready(&suggestion.command));
    } else if json {
        println!("{}", response::ready(&suggestion, &source, documentation_intent.as_deref())?);
    } else {
        println!("{}", suggestion.command);
    }
    Ok(())
}

#[cfg(feature = "model")]
#[derive(Clone, Copy, Eq, PartialEq)]
enum SuggestionOrigin {
    Model,
    Documentation,
}

#[cfg(feature = "model")]
#[allow(clippy::too_many_arguments)]
fn try_model_suggestion(
    index: &Index,
    endpoint: &str,
    context: &str,
    instruction: &str,
    platform: &str,
    prompt_contract: model::PromptContract,
    timeout_ms: u64,
    shortlist: &[String],
) -> Result<model::Suggestion> {
    let deadline = std::time::Instant::now()
        .checked_add(std::time::Duration::from_millis(timeout_ms))
        .context("--timeout-ms is too large")?;
    let suggestion = if context.trim().is_empty() {
        if shortlist.is_empty() {
            anyhow::bail!(
                "no installed command documentation matched the instruction; run `shelliq index scan` or pass explicit `--context`"
            );
        }
        let draft_context = shortlist_context(shortlist);
        let draft = model::suggest(
            endpoint,
            platform,
            &draft_context,
            instruction,
            prompt_contract,
            remaining(deadline)?,
        )?;
        enforce_shortlist(&draft.command_name, shortlist)?;
        let flags = index.search_flags(&draft.command_name, instruction, 16)?;
        let evidence = evidence_context(&draft.command_name, &flags);
        let final_suggestion = model::suggest(
            endpoint,
            platform,
            &evidence,
            instruction,
            prompt_contract,
            remaining(deadline)?,
        )?;
        enforce_selected_command(&final_suggestion.command_name, &draft.command_name)?;
        final_suggestion
    } else {
        model::suggest(
            endpoint,
            platform,
            context,
            instruction,
            prompt_contract,
            remaining(deadline)?,
        )?
    };
    let findings = shelliq_verify::verify(index, &suggestion.command)?;
    let failures: Vec<_> = findings.iter().filter(|finding| !finding.is_clean()).collect();
    if !failures.is_empty() {
        for finding in failures {
            eprintln!("{}", render::finding(finding));
        }
        anyhow::bail!("model suggestion failed local command/flag validation");
    }
    Ok(suggestion)
}

#[cfg(feature = "model")]
fn remaining(deadline: std::time::Instant) -> Result<std::time::Duration> {
    deadline
        .checked_duration_since(std::time::Instant::now())
        .filter(|duration| !duration.is_zero())
        .context("model request deadline expired")
}

#[cfg(feature = "model")]
fn shortlist_context(commands: &[String]) -> String {
    format!(
        "Selection pass: choose exactly one command name from this installed, indexed shortlist. Do not add flags yet.\nCommands:\n{}",
        commands.iter().map(|name| format!("- {name}")).collect::<Vec<_>>().join("\n")
    )
}

#[cfg(feature = "model")]
fn enforce_shortlist(command: &str, shortlist: &[String]) -> Result<()> {
    if shortlist.iter().any(|candidate| candidate == command) {
        return Ok(());
    }
    anyhow::bail!(
        "model selected `{command}`, which is outside the installed-command shortlist: {}",
        shortlist.join(", ")
    )
}

#[cfg(feature = "model")]
fn enforce_selected_command(command: &str, selected: &str) -> Result<()> {
    if command == selected {
        return Ok(());
    }
    anyhow::bail!("final model pass changed selected command from `{selected}` to `{command}`")
}

#[cfg(feature = "model")]
fn evidence_context(command: &str, flags: &[shelliq_index::FlagRow]) -> String {
    let mut lines = vec![format!(
        "Final pass: use exactly command `{command}`. Option tokens are case-sensitive. Use only exact option tokens listed below; every unlisted spelling or case variant is forbidden. Operands must come from the instruction."
    )];
    if flags.is_empty() {
        lines.push("No relevant options were retrieved; emit the command without options.".to_string());
    } else {
        lines.push("Authorized local options:".to_string());
        lines.extend(flags.iter().map(|flag| {
            let tokens = [flag.short.as_deref(), flag.long.as_deref()]
                .into_iter()
                .flatten()
                .map(|token| format!("`{token}`"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "- exact tokens: {tokens}; meaning: {}; citation: {}",
                flag.description,
                flag.citation()
            )
        }));
    }
    lines.join("\n")
}

fn scan(path: &std::path::Path) -> Result<()> {
    let names = shelliq_harvest::discover_path_commands();
    let mut index = Index::open(path)?;
    let mut commands = 0usize;
    let mut pages = 0usize;
    let mut flags = 0usize;

    for name in &names {
        let Ok(parsed) = shelliq_harvest::harvest_all(name) else {
            continue;
        };
        let target = shelliq_harvest::resolve_target(name, true);
        for command in parsed {
            flags += command.flags.len();
            pages += 1;
            index.insert_command(&target, &command)?;
        }
        index.insert_tldr_examples(name)?;
        commands += 1;
    }

    println!(
        "discovered {} commands; indexed {commands} with {pages} man pages and {flags} flags -> {}",
        names.len(),
        path.display()
    );
    shelliq_index::secure_permissions(path);
    Ok(())
}

fn build(path: &std::path::Path, names: &[String], allow_writable_paths: &[std::path::PathBuf]) -> Result<()> {
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
                index.insert_tldr_examples(name)?;
            }
            Err(man_err) => match harvest_via_help(&mut index, &target, name, allow_writable_paths) {
                Ok(flag_count) => {
                    pages += 1;
                    flags += flag_count;
                    index.insert_tldr_examples(name)?;
                    println!("  {name}(--help)  {flag_count} flags");
                }
                Err(help_err) => eprintln!("  {name}: no man page ({man_err}); --help crawl failed too: {help_err}"),
            },
        }
    }

    println!("\nindexed {pages} pages, {flags} flags -> {}", path.display());
    shelliq_index::secure_permissions(path);
    Ok(())
}

/// Fall back to crawling `--help` when a name has no man page, for tools like `ollama`,
/// `kubectl`, `cargo`, and `uv` that document themselves that way instead.
fn harvest_via_help(
    index: &mut Index,
    target: &shelliq_harvest::Target,
    name: &str,
    allow_writable_paths: &[std::path::PathBuf],
) -> Result<usize> {
    let exec_path = target
        .exec_path
        .as_deref()
        .context("no executable file to crawl via --help")?;
    let limits = shelliq_harvest::help_crawler::CrawlLimits {
        allow_paths: allow_writable_paths.to_vec(),
        ..Default::default()
    };
    let nodes = shelliq_harvest::help_crawler::crawl_help(std::path::Path::new(exec_path), target.exec_hash.as_deref(), &limits)?;
    let flag_count: usize = nodes.iter().map(|n| n.flags.len()).sum();
    index.insert_help_crawl(target, name, &nodes)?;
    Ok(flag_count)
}

/// Rebuild every harvested name into a sibling file, then swap it in atomically.
///
/// A refresh in place would leave a reader briefly looking at a half-rewritten index; this
/// instead only ever replaces the whole file in one `rename`, and a name that no longer
/// resolves to anything simply harvests nothing and drops out of the fresh index, so a
/// removed command's stale facts do not linger.
fn refresh(path: &std::path::Path, names: &[String], allow_writable_paths: &[std::path::PathBuf]) -> Result<()> {
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
                fresh.insert_tldr_examples(name)?;
            }
            Err(man_err) => match harvest_via_help(&mut fresh, &target, name, allow_writable_paths) {
                Ok(_) => {
                    pages += 1;
                    fresh.insert_tldr_examples(name)?;
                }
                Err(help_err) => eprintln!("  {name}: no man page ({man_err}); --help crawl failed too: {help_err}"),
            },
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
        bail_not_indexed(&index, command)?;
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

fn list_flags(path: &std::path::Path, command: &str, raw: bool) -> Result<()> {
    let index = Index::open(path)?;
    let flags = index.flags_for(command)?;

    if raw {
        for flag in &flags {
            if let Some(s) = &flag.short {
                println!("{s}");
            }
            if let Some(l) = &flag.long {
                println!("{l}");
            }
        }
        return Ok(());
    }

    warn_if_stale(&index, command)?;
    if flags.is_empty() {
        bail_not_indexed(&index, command)?;
    }
    for flag in &flags {
        println!("{}", render::flag_line(flag));
    }
    println!("\n{} flags", flags.len());
    Ok(())
}

/// `flags_for` returns nothing both when a name was never harvested and when it was
/// harvested but has no top-level flags of its own — real for tools like `kubectl`, whose
/// `--help` lists only subcommands, with flags living entirely under them. Telling those
/// apart matters: the first case is fixed by `index build`, the second is not.
fn bail_not_indexed(index: &Index, command: &str) -> Result<()> {
    if index.command_exists(command)? {
        anyhow::bail!(
            "`{command}` is indexed but has no top-level flags of its own; \
             they live under its subcommands, which `shelliq flags`/`search` cannot query yet"
        );
    }
    anyhow::bail!("`{command}` is not indexed; run `shelliq index build {command}`");
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

#[cfg(all(test, feature = "model"))]
mod tests {
    use super::*;

    #[test]
    fn parses_index_scan_subcommand() {
        let cli = Cli::try_parse_from(["shelliq", "index", "scan"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Index {
                action: IndexAction::Scan
            }
        ));
    }

    #[test]
    fn zsh_widget_protocol_conflicts_with_json_output() {
        let parsed = Cli::try_parse_from(["shelliq", "suggest", "--zsh-widget", "--json", "find files"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_a_draft_command_outside_the_shortlist() {
        let shortlist = vec!["cp".to_string(), "rsync".to_string()];
        enforce_shortlist("cp", &shortlist).unwrap();
        let error = enforce_shortlist("curl", &shortlist).unwrap_err();
        assert!(error.to_string().contains("outside the installed-command shortlist"));
    }

    #[test]
    fn final_pass_cannot_change_the_selected_command() {
        enforce_selected_command("cp", "cp").unwrap();
        let error = enforce_selected_command("rsync", "cp").unwrap_err();
        assert!(error.to_string().contains("changed selected command"));
    }

    #[test]
    fn evidence_lists_exact_case_sensitive_option_tokens() {
        let flag = shelliq_index::FlagRow {
            command: "cp".into(),
            section: "1".into(),
            short: Some("-R".into()),
            long: Some("--recursive".into()),
            arg_type: None,
            arg_required: false,
            description: "copy directories recursively".into(),
            flag_group: None,
            source_line: 66,
            excerpt: String::new(),
            rank_personal: 0,
        };
        let context = evidence_context("cp", &[flag]);
        assert!(context.contains("Option tokens are case-sensitive"));
        assert!(context.contains("exact tokens: `-R`, `--recursive`"));
        assert!(!context.contains("`-r`"));
    }
}
