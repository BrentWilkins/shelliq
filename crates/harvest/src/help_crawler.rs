//! Harvesting command facts from `--help` output, for tools with no man page.
//!
//! `--help` is a convention, not a contract: a binary is free to load plugins, read
//! credentials, open sockets, write files, or fork before it ever looks at argv, and
//! reaching a subcommand's own `--help` means passing that subcommand too. See PLAN.md's
//! "Containment, honestly" table — every row there has a matching control below, with one
//! documented gap: **fork bombs are not contained**. `RLIMIT_NPROC` looked like the answer
//! but is a per-*uid* limit on Linux, not a per-process-tree one — setting it low in the
//! child collides with however many processes the user's desktop session already has
//! running and starts killing unrelated work, which was verified here by crashing a live
//! `cargo` build. The real control is a pids cgroup, which needs a delegated cgroup
//! hierarchy this crate does not assume exists; until that lands, a forking `--help`
//! implementation is only bounded by the timeout, same as the P0 answer PLAN.md already
//! called "consent, not containment."

use crate::{ParsedFlag, hex, parse_spec, push_unique, split_tag, strip_control_chars};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Bounds on one crawl, enforced independently of any single tool's behaviour.
#[derive(Debug, Clone)]
pub struct CrawlLimits {
    /// How many levels of subcommand to recurse into: `ollama` -> `ollama list` -> depth 1.
    pub max_depth: usize,
    /// Killed, group and all, if a single `--help` invocation runs longer than this.
    pub timeout: Duration,
    /// Hard cap on captured stdout+stderr per invocation.
    pub max_output_bytes: usize,
    /// Hard cap on total `--help` invocations across the whole crawl, so a tool cannot
    /// advertise its way into unbounded forking through a wide subcommand tree.
    pub max_nodes: usize,
    /// Directories a caller has explicitly approved for execution even though they are
    /// writable by someone other than root — e.g. `~/.cargo/bin`. Empty by default: nothing
    /// outside a root-owned, non-writable directory runs unless opted in here.
    pub allow_paths: Vec<PathBuf>,
}

impl Default for CrawlLimits {
    fn default() -> Self {
        CrawlLimits {
            max_depth: 3,
            timeout: Duration::from_secs(5),
            max_output_bytes: 1 << 20,
            max_nodes: 200,
            allow_paths: Vec::new(),
        }
    }
}

/// One `--help` invocation's parsed facts.
#[derive(Debug, Clone)]
pub struct HelpNode {
    /// Subcommand path that reached this invocation, e.g. `["list"]` for `ollama list --help`.
    pub path: Vec<String>,
    pub flags: Vec<ParsedFlag>,
    /// Further subcommands this node advertises, for the caller to see what was (or, past
    /// `max_depth` or `max_nodes`, was not) followed.
    pub subcommands: Vec<String>,
}

/// Crawl `exec_path --help`, then each advertised subcommand's `--help`, breadth-first to
/// `limits.max_depth`.
///
/// `expected_hash`, when given, is compared against the file's content immediately before
/// every exec, not just once at the start — an approval made at the top of the crawl must
/// still hold at the bottom of it.
pub fn crawl_help(exec_path: &Path, expected_hash: Option<&str>, limits: &CrawlLimits) -> Result<Vec<HelpNode>> {
    let mut out = Vec::new();
    let mut queue: VecDeque<Vec<String>> = VecDeque::new();
    queue.push_back(Vec::new());
    let mut visited: HashSet<Vec<String>> = HashSet::new();

    while let Some(path) = queue.pop_front() {
        if out.len() >= limits.max_nodes || !visited.insert(path.clone()) {
            continue;
        }

        let mut args = path.clone();
        args.push("--help".to_string());
        let bytes = run_help_contained(exec_path, expected_hash, &args, limits)
            .with_context(|| format!("running `{} {}`", exec_path.display(), args.join(" ")))?;
        let text = strip_control_chars(&String::from_utf8_lossy(&bytes));
        let (flags, subcommands) = parse_help_text(&text);

        if path.len() < limits.max_depth {
            for sub in &subcommands {
                let mut next = path.clone();
                next.push(sub.clone());
                queue.push_back(next);
            }
        }

        out.push(HelpNode {
            path,
            flags,
            subcommands,
        });
    }

    Ok(out)
}

/// Run one `exec_path args...` under containment and return captured output.
pub fn run_help_contained(
    exec_path: &Path,
    expected_hash: Option<&str>,
    args: &[String],
    limits: &CrawlLimits,
) -> Result<Vec<u8>> {
    let real = reverify_target(exec_path, expected_hash, limits)?;

    let scratch = ScratchDir::new().context("creating a scratch directory for the crawl")?;
    // Exec through `exec_path`, not the canonicalized `real`, so a multi-call binary that
    // dispatches on argv[0] (rustup's `cargo` proxy, busybox-style tools) sees the name it
    // was actually invoked as, exactly as a shell's own PATH lookup would leave it.
    let mut cmd = Command::new(exec_path);
    cmd.args(args)
        .current_dir(scratch.path())
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", scratch.path())
        .env("LANG", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // SAFETY: the closure only calls the async-signal-safe libc function setsid between
    // fork and exec, and touches no Rust-managed state.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn().context("spawning --help crawl")?;
    let pid = child.id() as libc::pid_t;

    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let max_bytes = limits.max_output_bytes;
    let out_reader = std::thread::spawn(move || read_capped(&mut stdout, max_bytes));
    let err_reader = std::thread::spawn(move || read_capped(&mut stderr, max_bytes));

    let deadline = Instant::now() + limits.timeout;
    let timed_out = loop {
        if child.try_wait().context("polling --help crawl")?.is_some() {
            break false;
        }
        if Instant::now() >= deadline {
            break true;
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    if timed_out {
        // Kill the whole session, not just the direct child, so descendants spawned
        // before the timeout don't survive the deadline. This has to happen before
        // `wait()`, since a hung child would otherwise never let it return.
        unsafe { libc::kill(-pid, libc::SIGKILL) };
    }
    let _ = child.wait();
    // Sweep the process group unconditionally, not only on timeout: a `--help` that exits
    // quickly while a background process it spawned lingers must not escape the crawl's
    // boundary either. `setsid` made this pgid ours alone, and a pgid remains signalable
    // for as long as any of its members survive, even after the leader that created it
    // has already exited.
    unsafe { libc::kill(-pid, libc::SIGKILL) };

    let out = out_reader.join().unwrap_or_default();
    let err = err_reader.join().unwrap_or_default();
    if timed_out && out.is_empty() && err.is_empty() {
        bail!("`{}` did not finish `--help` within {:?}", real.display(), limits.timeout);
    }
    Ok(if out.is_empty() { err } else { out })
}

fn read_capped(r: &mut impl Read, cap: usize) -> Vec<u8> {
    let mut chunk = [0u8; 64 * 1024];
    let mut out = Vec::new();
    while out.len() < cap {
        match r.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => out.extend_from_slice(&chunk[..n]),
        }
    }
    out.truncate(cap);
    out
}

/// Re-resolve, re-stat, and (if given) re-hash `exec_path` immediately before it is run.
///
/// An allowlist checked once at approval time and trusted afterward is a TOCTOU bug: the
/// name could resolve somewhere else, or the file's content could change, between approval
/// and exec. Returns the canonical path actually about to be executed.
fn reverify_target(exec_path: &Path, expected_hash: Option<&str>, limits: &CrawlLimits) -> Result<PathBuf> {
    let real = std::fs::canonicalize(exec_path).with_context(|| format!("resolving `{}`", exec_path.display()))?;
    let meta = std::fs::metadata(&real).with_context(|| format!("stat `{}`", real.display()))?;
    if !meta.is_file() {
        bail!("`{}` is not a regular file", real.display());
    }

    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let mode = meta.permissions().mode();
    if mode & 0o6000 != 0 {
        bail!("`{}` is setuid or setgid; refusing to execute it", real.display());
    }

    if let Some(expected) = expected_hash {
        let bytes = std::fs::read(&real).with_context(|| format!("reading `{}`", real.display()))?;
        let actual = hex(&Sha256::digest(&bytes));
        if actual != expected {
            bail!(
                "`{}` no longer matches the hash it was approved under; re-approve before crawling it",
                real.display()
            );
        }
    }

    if let Some(dir) = real.parent() {
        let dir_meta = std::fs::metadata(dir).with_context(|| format!("stat `{}`", dir.display()))?;
        let root_owned_and_locked_down = dir_meta.uid() == 0 && dir_meta.permissions().mode() & 0o022 == 0;
        let opted_in = limits.allow_paths.iter().any(|p| real.starts_with(p));
        if !root_owned_and_locked_down && !opted_in {
            bail!(
                "`{}` lives in `{}`, which is writable outside root; add it to `allow_paths` to opt in",
                real.display(),
                dir.display()
            );
        }
    }

    Ok(real)
}

/// A working directory that exists only for one crawl and is removed with it, so a `--help`
/// invocation never sees the caller's actual `cwd`.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self> {
        let unique = format!(
            "shelliq-help-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir(&path).with_context(|| format!("creating `{}`", path.display()))?;
        Ok(ScratchDir(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Flags,
    Commands,
    Other,
}

fn classify_section(heading: &str) -> Section {
    let lower = heading.to_ascii_lowercase();
    if lower.contains("command") {
        Section::Commands
    } else if lower.contains("option") || lower.contains("flag") {
        Section::Flags
    } else {
        Section::Other
    }
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// Parse the near-universal `-x, --xxx  description` shape that clap, cobra, and argparse
/// all emit, plus each tool's own subcommand list, so `crawl_help` can recurse.
///
/// Unlike a man page, `--help` output carries no fixed column contract — every tool picks
/// its own indentation — so each section's item indent is discovered per tool rather than
/// assumed, and a flag's description may sit inline or wrap onto more-indented lines below
/// it, exactly as a man page's `.TP` body does.
pub fn parse_help_text(text: &str) -> (Vec<ParsedFlag>, Vec<String>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut flags: Vec<ParsedFlag> = Vec::new();
    let mut subcommands: Vec<String> = Vec::new();

    let mut section = Section::Other;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') && trimmed.ends_with(':') {
            section = classify_section(trimmed);
            i += 1;
            continue;
        }

        let indent = indent_of(line);
        let body = line[indent..].trim_end();

        if section == Section::Flags && body.starts_with('-') {
            let (spec, inline) = split_tag(body);
            let spec = spec.strip_suffix("...").unwrap_or(spec);
            if let Some(mut parsed) = parse_spec(spec) {
                let mut desc_parts: Vec<String> = Vec::new();
                if !inline.trim().is_empty() {
                    desc_parts.push(inline.trim().to_string());
                }
                // A continuation line is anything that isn't itself a new flag or blank.
                // Indent alone cannot mark the boundary: a long-only flag like `--color`
                // is indented *past* a short one like `-q` to keep names aligned, so a
                // purely indent-based check would swallow it into the previous
                // description instead of starting a new flag.
                let mut j = i + 1;
                while j < lines.len() {
                    let b = lines[j];
                    if b.trim().is_empty() || b.trim_start().starts_with('-') {
                        break;
                    }
                    desc_parts.push(b.trim().to_string());
                    j += 1;
                }
                parsed.description = strip_control_chars(&desc_parts.join(" "));
                push_unique(&mut flags, parsed);
                i = j;
                continue;
            }
        } else if section == Section::Commands
            && !body.starts_with('-')
            && let Some(name) = subcommand_name(body)
        {
            subcommands.push(name);
        }

        i += 1;
    }

    (flags, subcommands)
}

/// First name on a commands-section line, e.g. `build, b    Compile...` -> `build`, or
/// `create          Create a resource...` -> `create`. Rejects continuation markers like
/// cargo's `...  See all commands with --list`, which is not a real subcommand name.
fn subcommand_name(body: &str) -> Option<String> {
    let first = body.split(|c: char| c.is_whitespace() || c == ',').next()?.trim();
    let is_name = !first.is_empty() && first.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    is_name.then(|| first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cargo_style_inline_descriptions_and_commands() {
        let text = "\
Usage: cargo [OPTIONS] [COMMAND]

Options:
  -V, --version                  Print version info and exit
  -v, --verbose...               Use verbose output

Commands:
    build, b    Compile the current package
    ...         See all commands with --list
";
        let (flags, subcommands) = parse_help_text(text);
        assert_eq!(flags.len(), 2);
        assert_eq!(flags[0].long.as_deref(), Some("--version"));
        assert_eq!(flags[1].long.as_deref(), Some("--verbose"));
        assert_eq!(subcommands, vec!["build".to_string()]);
    }

    #[test]
    fn parses_wrapped_description_below_the_flag_line() {
        let text = "\
Global options:
  -q, --quiet...
          Use quiet output
      --offline
          Disable network access
";
        let (flags, _) = parse_help_text(text);
        assert_eq!(flags.len(), 2);
        assert_eq!(flags[0].short.as_deref(), Some("-q"));
        assert_eq!(flags[0].description, "Use quiet output");
        assert_eq!(flags[1].long.as_deref(), Some("--offline"));
        assert_eq!(flags[1].description, "Disable network access");
    }

    #[test]
    fn parses_cobra_available_commands_section() {
        let text = "\
Available Commands:
  serve        Start Ollama
  list         List models

Flags:
  -h, --help         help for ollama
      --nowordwrap   Don't wrap words to the next line automatically
";
        let (flags, subcommands) = parse_help_text(text);
        assert_eq!(subcommands, vec!["serve".to_string(), "list".to_string()]);
        assert_eq!(flags.len(), 2);
        assert_eq!(flags[1].long.as_deref(), Some("--nowordwrap"));
    }
}
