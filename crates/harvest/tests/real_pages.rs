//! Parser accuracy against the man pages actually installed on this machine.
//!
//! Ground truth is each tool's own `--help`, which is authoritative in a way that a
//! hand-counted figure is not. The counts quoted during planning (`curl` 396, `rsync` 382)
//! turned out to be an overcount: that heuristic also matched flags *mentioned* inside
//! description bodies, such as curl's "--alt-svc can be used several times". These tests
//! record the measured truth instead.
//!
//! Every test skips rather than fails when the tool or its man page is absent, so the
//! suite stays honest on a machine with a different set of packages.

use std::collections::BTreeSet;
use std::process::Command;

fn have_page(name: &str) -> bool {
    shelliq_harvest::man_paths(name)
        .map(|paths| !paths.is_empty())
        .unwrap_or(false)
}

/// Long flags as the tool's own `--help` reports them.
fn help_flags(cmd: &str, args: &[&str]) -> Option<BTreeSet<String>> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let take = |s: &str| -> Option<String> {
        let t = s.trim_start();
        if !t.starts_with("--") {
            return None;
        }
        let n: String = t
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '.')
            .collect();
        (n.len() > 2).then_some(n)
    };
    Some(
        text.lines()
            .filter(|l| l.starts_with(' '))
            .filter_map(|l| {
                let t = l.trim_start();
                if t.starts_with("--") {
                    take(t)
                } else if t.len() > 4 && t.starts_with('-') {
                    take(&t[4..])
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn parsed_long_flags(name: &str) -> BTreeSet<String> {
    shelliq_harvest::harvest(name)
        .expect("harvest")
        .flags
        .into_iter()
        .filter_map(|f| f.long)
        .collect()
}

/// curl has the most flags of anything installed and a strictly formatted `--help all`,
/// which makes it the sharpest available test of the parser.
#[test]
fn curl_matches_its_help_or_the_known_apple_skew() {
    if !have_page("curl") {
        eprintln!("skipping: no curl man page");
        return;
    }
    let Some(truth) = help_flags("curl", &["--help", "all"]) else {
        eprintln!("skipping: curl not runnable");
        return;
    };
    let parsed = parsed_long_flags("curl");

    let only_man: Vec<_> = parsed.difference(&truth).collect();
    let only_help: Vec<_> = truth.difference(&parsed).collect();
    #[cfg(target_os = "macos")]
    {
        // Apple's curl man page can lead its bundled executable by these curl
        // 8.3 expansion options. Continue rejecting every other divergence.
        let apple_man_only = ["--expand-data", "--expand-url", "--expand-variable"];
        assert!(
            only_help.is_empty() && only_man.iter().all(|flag| apple_man_only.contains(&flag.as_str())),
            "curl flag sets diverge\n  only in man:  {only_man:?}\n  only in help: {only_help:?}"
        );
    }
    #[cfg(not(target_os = "macos"))]
    assert!(
        only_man.is_empty() && only_help.is_empty(),
        "curl flag sets diverge\n  only in man:  {only_man:?}\n  only in help: {only_help:?}"
    );
    assert!(parsed.len() > 200, "expected a large flag set, got {}", parsed.len());
}

#[test]
fn ls_matches_its_own_help_exactly() {
    if !have_page("ls") {
        eprintln!("skipping: no ls man page");
        return;
    }
    let Some(truth) = help_flags("ls", &["--help"]) else {
        return;
    };
    let parsed = parsed_long_flags("ls");
    assert_eq!(
        parsed,
        truth,
        "ls flag sets diverge\n  only in man:  {:?}\n  only in help: {:?}",
        parsed.difference(&truth).collect::<Vec<_>>(),
        truth.difference(&parsed).collect::<Vec<_>>()
    );
}

/// Both case-sensitive spellings at the heart of the original complaint must
/// reflect the documentation installed on this platform.
#[test]
fn grep_r_and_capital_r_are_indexed_as_documented() {
    if !have_page("grep") {
        eprintln!("skipping: no grep man page");
        return;
    }
    let cmd = shelliq_harvest::harvest("grep").expect("harvest grep");
    let find = |s: &str| cmd.flags.iter().find(|f| f.short.as_deref() == Some(s));

    let spellings = || cmd.flags.iter().map(|flag| flag.spelling()).collect::<Vec<_>>();
    let lower = find("-r").unwrap_or_else(|| panic!("grep -r must be indexed; harvested spellings: {:?}", spellings()));
    let upper = find("-R").unwrap_or_else(|| panic!("grep -R must be indexed; harvested spellings: {:?}", spellings()));

    #[cfg(not(target_os = "macos"))]
    {
        assert_eq!(lower.long.as_deref(), Some("--recursive"));
        assert_eq!(upper.long.as_deref(), Some("--dereference-recursive"));
        assert_ne!(lower.description, upper.description);
    }
    assert!(lower.source_line > 0 && upper.source_line > 0, "citations required");
}

/// Multiple long spellings grouped on one man-page tag must all survive parsing.
#[test]
fn grouped_grep_aliases_cover_its_help() {
    if !have_page("grep") {
        eprintln!("skipping: no grep man page");
        return;
    }
    let Some(truth) = help_flags("grep", &["--help"]) else {
        return;
    };
    let parsed = parsed_long_flags("grep");
    let missing_from_man: Vec<_> = truth.difference(&parsed).collect();
    assert!(
        missing_from_man.is_empty(),
        "grep help flags missing from parsed grouped man-page aliases: {missing_from_man:?}"
    );
}

/// `man -w` returns only the default section, which loses real content.
///
/// On this machine `signal` defaults to the 84-line section 2 syscall page while the
/// 378-line section 7 overview is the one a user usually wants. Harvesting must cover
/// every section for a name.
#[test]
fn every_section_is_harvested_not_just_the_default() {
    let Ok(paths) = shelliq_harvest::man_paths("signal") else {
        eprintln!("skipping: no signal man page");
        return;
    };
    if paths.len() < 2 {
        eprintln!("skipping: signal only documented in one section here");
        return;
    }
    let all = shelliq_harvest::harvest_all("signal").expect("harvest_all signal");
    let sections: BTreeSet<String> = all.iter().map(|c| c.section.clone()).collect();
    assert!(
        sections.contains("7"),
        "section 7 signal overview must be indexed, got sections {sections:?}"
    );
    assert!(sections.len() >= 2, "expected multiple sections, got {sections:?}");
}

/// Section preference decides which page answers a bare command lookup.
#[test]
fn command_sections_outrank_syscall_sections() {
    assert!(shelliq_harvest::section_rank("1") < shelliq_harvest::section_rank("2"));
    assert!(shelliq_harvest::section_rank("8") < shelliq_harvest::section_rank("3"));

    if shelliq_harvest::man_paths("kill").map(|p| p.len()).unwrap_or(0) < 2 {
        eprintln!("skipping: kill not multi-section here");
        return;
    }
    // `kill` is a shell command in section 1 and a syscall in section 2.
    let cmd = shelliq_harvest::harvest("kill").expect("harvest kill");
    assert_eq!(cmd.section, "1", "bare `kill` must resolve to the command");
}

#[test]
fn descriptions_and_citations_are_populated() {
    if !have_page("ls") {
        eprintln!("skipping: no ls man page");
        return;
    }
    let cmd = shelliq_harvest::harvest("ls").expect("harvest ls");
    let described = cmd.flags.iter().filter(|f| !f.description.is_empty()).count();
    assert!(
        described * 10 >= cmd.flags.len() * 9,
        "at least 90% of flags should carry a description, got {described}/{}",
        cmd.flags.len()
    );
    assert!(cmd.flags.iter().all(|f| f.source_line > 0));
    assert_eq!(cmd.section, "1");
}
