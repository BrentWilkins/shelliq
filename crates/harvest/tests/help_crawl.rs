//! `--help` crawler exercised against tools actually installed on this machine, and against
//! a deliberately hostile fixture binary this suite writes for itself.

use shelliq_harvest::help_crawler::{CrawlLimits, crawl_help, run_help_contained};
use std::path::{Path, PathBuf};

fn find_on_path(name: &str) -> Option<PathBuf> {
    let target = shelliq_harvest::resolve_target(name, false);
    target.exec_path.map(PathBuf::from)
}

fn limits_allowing(path: &std::path::Path) -> CrawlLimits {
    let mut limits = CrawlLimits::default();
    if let Some(parent) = path.parent() {
        limits.allow_paths.push(parent.to_path_buf());
    }
    limits
}

/// `cargo` on this machine is rustup's toolchain-selecting proxy, which needs `$HOME` to
/// find its own config and so cannot be exercised end to end under the crawler's stripped
/// environment. `kubectl` is a plain binary with no such indirection. Its
/// installation directory is explicitly opted in because local ownership varies.
fn kubectl() -> Option<PathBuf> {
    let path = PathBuf::from("/usr/local/bin/kubectl");
    path.is_file().then_some(path)
}

#[test]
fn crawls_kubectl_flags_and_subcommands() {
    let Some(path) = kubectl() else { return };
    let limits = limits_allowing(&path);
    let nodes = crawl_help(&path, None, &limits).expect("crawl kubectl --help");

    let root = nodes.iter().find(|n| n.path.is_empty()).expect("root node");
    assert!(root.subcommands.contains(&"create".to_string()));

    let get = nodes.iter().find(|n| n.path == vec!["get".to_string()]);
    assert!(get.is_some(), "expected `kubectl get --help` to be crawled");
}

#[test]
fn refuses_paths_not_opted_in() {
    let Some(path) = find_on_path("uv") else { return };
    let limits = CrawlLimits::default();
    let err = crawl_help(&path, None, &limits).unwrap_err();
    assert!(format!("{err:#}").contains("allow_paths"), "{err:#}");
}

#[test]
fn crawls_an_opted_in_writable_path() {
    let Some(path) = find_on_path("uv") else { return };
    let limits = limits_allowing(&path);
    let nodes = crawl_help(&path, None, &limits).expect("crawl uv --help");
    let root = nodes.iter().find(|n| n.path.is_empty()).expect("root node");
    assert!(root.flags.iter().any(|f| f.long.as_deref() == Some("--version")));
}

#[test]
fn rejects_a_hash_mismatch() {
    let Some(path) = kubectl() else { return };
    let limits = CrawlLimits::default();
    let bogus = "0".repeat(64);
    let err = crawl_help(&path, Some(&bogus), &limits).unwrap_err();
    assert!(format!("{err:#}").contains("no longer matches"), "{err:#}");
}

/// A `--help` implementation, written by this suite, that forks, writes, floods stdout, and
/// tries to read stdin — the exact hostile behaviours PLAN.md's containment table promises
/// to survive. Its directory is opted into `allow_paths` deliberately, the way a caller
/// would for a real writable-by-non-root tool; the containment being tested here is what
/// happens once execution is already approved, not the approval check itself (covered by
/// `refuses_paths_not_opted_in`).
struct HostileScript(PathBuf);

impl HostileScript {
    fn new(name: &str, body: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("shelliq-hostile-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create hostile fixture dir");
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write hostile fixture");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).expect("chmod hostile fixture");
        HostileScript(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn limits(&self) -> CrawlLimits {
        let mut limits = CrawlLimits::default();
        limits.allow_paths.push(self.0.parent().unwrap().to_path_buf());
        limits
    }
}

impl Drop for HostileScript {
    fn drop(&mut self) {
        if let Some(dir) = self.0.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn pid_alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid])
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn resolves_an_opted_in_directory_before_checking_containment() {
    let script = HostileScript::new("canonical_path.sh", "echo done");
    let alias = std::env::temp_dir().join(format!("shelliq-hostile-{}-canonical-alias", std::process::id()));
    std::os::unix::fs::symlink(script.path().parent().unwrap(), &alias).expect("create fixture directory symlink");

    let alias_script = alias.join(script.path().file_name().unwrap());
    let mut limits = CrawlLimits::default();
    limits.allow_paths.push(alias.clone());
    let result = run_help_contained(&alias_script, None, &["--help".to_string()], &limits);

    std::fs::remove_file(&alias).expect("remove fixture directory symlink");
    let bytes = result.expect("crawl through an explicitly approved directory symlink");
    assert_eq!(String::from_utf8_lossy(&bytes).trim(), "done");
}

#[test]
fn a_quickly_exiting_hostile_binary_does_not_leave_its_forked_child_running() {
    let script = HostileScript::new(
        "forks_and_exits.sh",
        r#"( i=0; while [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done ) >/dev/null 2>&1 &
echo "HOSTILE_CHILD_PID=$!"
echo "left evidence" > evidence.txt
exit 0"#,
    );
    let limits = script.limits();
    let bytes =
        run_help_contained(script.path(), None, &["--help".to_string()], &limits).expect("hostile script runs to completion");
    let out = String::from_utf8_lossy(&bytes);
    let pid = out
        .lines()
        .find_map(|l| l.strip_prefix("HOSTILE_CHILD_PID="))
        .expect("hostile script reports its forked child's pid");

    assert!(
        !Path::new("evidence.txt").exists(),
        "the hostile write must land in the crawl's scratch cwd, not the caller's"
    );

    // `--help` returning is not proof the fork was reaped; give the sweep a moment to land.
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !pid_alive(pid),
        "forked child `{pid}` outlived a --help that exited on its own"
    );
}

#[test]
fn output_flooding_past_the_cap_is_killed_within_the_timeout_not_left_to_hang() {
    let script = HostileScript::new(
        "floods_forever.sh",
        "echo \"Options:\"\necho \"  -x, --xxx  a flooding flag\"\nyes AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    );
    let mut limits = script.limits();
    limits.timeout = std::time::Duration::from_millis(300);
    limits.max_output_bytes = 4096;

    let started = std::time::Instant::now();
    let result = run_help_contained(script.path(), None, &["--help".to_string()], &limits);
    let elapsed = started.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "a flooding, non-terminating --help hung well past its {:?} timeout: {elapsed:?}",
        limits.timeout
    );
    let bytes = result.expect("the flag line was captured before the cap silenced the flood");
    assert!(bytes.len() <= limits.max_output_bytes);
    assert!(String::from_utf8_lossy(&bytes).contains("--xxx"));
}

#[test]
fn a_hostile_binary_reading_stdin_gets_immediate_eof_not_a_hang() {
    let script = HostileScript::new("reads_stdin.sh", "cat >/dev/null\necho done");
    let limits = script.limits();

    let started = std::time::Instant::now();
    let bytes = run_help_contained(script.path(), None, &["--help".to_string()], &limits)
        .expect("cat on a closed stdin returns immediately");
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
    assert!(String::from_utf8_lossy(&bytes).contains("done"));
}

#[test]
fn ansi_escape_sequences_in_help_output_never_reach_a_stored_flag_description() {
    let script = HostileScript::new(
        "escapes.sh",
        "printf 'Options:\\n'\nprintf '  -x, --xxx  \\033[31mRed\\033[0m warning text\\n'",
    );
    let limits = script.limits();
    let nodes = crawl_help(script.path(), None, &limits).expect("crawl the hostile script");
    let root = nodes.iter().find(|n| n.path.is_empty()).expect("root node");
    let flag = root
        .flags
        .iter()
        .find(|f| f.long.as_deref() == Some("--xxx"))
        .expect("--xxx parsed");

    // `strip_control_chars` removes the raw ESC byte that makes a CSI sequence dangerous to
    // print, not the printable digits and brackets around it — that's the actual guarantee
    // (no control byte reaches a terminal), not a claim of cosmetically clean text.
    assert!(
        !flag.description.chars().any(|c| c.is_control() && c != '\t' && c != '\n'),
        "a raw control character reached a stored description: {:?}",
        flag.description
    );
    assert_eq!(flag.description, "[31mRed[0m warning text");
}
