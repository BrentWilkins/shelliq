//! The SQLite index of command facts.
//!
//! Everything shelliq states as fact — that `-r` exists, that `-R` means something else,
//! that `--block-size` takes an argument — is read from here and nowhere else. The model,
//! when there is one, never supplies a flag fact. That separation is what makes an answer
//! citable.
//!
//! Text comparison uses SQLite's default BINARY collation throughout, so `-r` and `-R` are
//! different rows and a lookup for one never returns the other.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use shelliq_harvest::{ParsedCommand, section_rank};

const SCHEMA: &str = include_str!("schema.sql");

/// One flag as stored, with enough context to cite it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlagRow {
    pub command: String,
    pub section: String,
    pub short: Option<String>,
    pub long: Option<String>,
    pub arg_type: Option<String>,
    pub arg_required: bool,
    pub description: String,
    pub flag_group: Option<String>,
    pub source_line: usize,
    pub rank_personal: i64,
}

impl FlagRow {
    /// `-r, --recursive`
    pub fn spelling(&self) -> String {
        match (&self.short, &self.long) {
            (Some(s), Some(l)) => format!("{s}, {l}"),
            (Some(s), None) => s.clone(),
            (None, Some(l)) => l.clone(),
            (None, None) => String::new(),
        }
    }

    /// `grep(1):168`
    pub fn citation(&self) -> String {
        format!("{}({}):{}", self.command, self.section, self.source_line)
    }
}

/// Result of checking one flag token against the index.
///
/// `CaseMismatch` is deliberately distinct from `Unknown`. Reporting "`-R` is not valid for
/// grep; did you mean `-r`?" is the entire point of the tool, and collapsing it into a
/// generic failure would throw away the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagLookup {
    Exact(Box<FlagRow>),
    CaseMismatch { typed: String, suggestion: Box<FlagRow> },
    Unknown { typed: String },
}

pub struct Index {
    conn: Connection,
}

impl Index {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path).context("opening index")?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA).context("applying schema")?;
        Ok(Self { conn })
    }

    /// Default index location, honouring `SHELLIQ_INDEX`.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("SHELLIQ_INDEX") {
            return p.into();
        }
        let base = std::env::var("XDG_DATA_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                    .join(".local/share")
            });
        base.join("shelliq/index.sqlite")
    }

    /// Insert or replace one command and all of its flags.
    ///
    /// Rows are replaced wholesale rather than merged, so a flag removed upstream
    /// disappears from the index instead of lingering as a fact that is no longer true.
    pub fn insert_command(&mut self, cmd: &ParsedCommand) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM commands WHERE name = ?1 AND platform = ?2 AND section = ?3",
            params![cmd.name, cmd.platform, cmd.section],
        )?;
        tx.execute(
            "INSERT INTO commands
                 (name, platform, section, synopsis, description,
                  source_path, source_hash, parser_version, harvested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now'))",
            params![
                cmd.name,
                cmd.platform,
                cmd.section,
                cmd.synopsis,
                cmd.description,
                cmd.source_path,
                cmd.source_hash,
                shelliq_harvest::PARSER_VERSION,
            ],
        )?;
        let command_id = tx.last_insert_rowid();

        {
            let mut stmt = tx.prepare(
                "INSERT INTO flags
                     (command_id, short, long, arg_type, arg_required,
                      description, flag_group, source_line)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for f in &cmd.flags {
                stmt.execute(params![
                    command_id,
                    f.short,
                    f.long,
                    f.arg_type,
                    f.arg_required as i64,
                    f.description,
                    f.group,
                    f.source_line as i64,
                ])?;
            }
        }

        tx.commit()?;
        Ok(command_id)
    }

    pub fn command_exists(&self, name: &str) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM commands WHERE name = ?1",
            params![name],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// All flags for a command, preferring the section a shell user means.
    pub fn flags_for(&self, command: &str) -> Result<Vec<FlagRow>> {
        let section = match self.preferred_section(command)? {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let mut stmt = self.conn.prepare(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.name = ?1 AND c.section = ?2
             ORDER BY f.rank_personal DESC, f.source_line",
        )?;
        let rows = stmt
            .query_map(params![command, section], row_to_flag)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The section that answers a bare command lookup, by `SECTION_PREFERENCE`.
    fn preferred_section(&self, command: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT section FROM commands WHERE name = ?1")?;
        let mut sections = stmt
            .query_map(params![command], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        sections.sort_by_key(|s| section_rank(s));
        Ok(sections.into_iter().next())
    }

    /// Check one flag token, case-sensitively, then case-insensitively as a suggestion.
    pub fn lookup_flag(&self, command: &str, token: &str) -> Result<FlagLookup> {
        let Some(section) = self.preferred_section(command)? else {
            return Ok(FlagLookup::Unknown { typed: token.to_string() });
        };

        let exact = self.query_one(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.name = ?1 AND c.section = ?2 AND (f.short = ?3 OR f.long = ?3)
             LIMIT 1",
            command,
            &section,
            token,
        )?;
        if let Some(row) = exact {
            return Ok(FlagLookup::Exact(Box::new(row)));
        }

        // `lower()` is ASCII-only in SQLite, which is correct here: flag names are ASCII.
        let folded = self.query_one(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.name = ?1 AND c.section = ?2
               AND (lower(f.short) = lower(?3) OR lower(f.long) = lower(?3))
             LIMIT 1",
            command,
            &section,
            token,
        )?;
        Ok(match folded {
            Some(row) => FlagLookup::CaseMismatch {
                typed: token.to_string(),
                suggestion: Box::new(row),
            },
            None => FlagLookup::Unknown { typed: token.to_string() },
        })
    }

    fn query_one(
        &self,
        sql: &str,
        command: &str,
        section: &str,
        token: &str,
    ) -> Result<Option<FlagRow>> {
        let mut stmt = self.conn.prepare(sql)?;
        let row = stmt
            .query_row(params![command, section, token], row_to_flag)
            .optional()?;
        Ok(row)
    }

    /// Full-text search over flag descriptions.
    ///
    /// This is the path that turns "follow redirect" into `-L, --location` without the
    /// user knowing the flag's name.
    pub fn search_flags(&self, command: &str, query: &str, limit: usize) -> Result<Vec<FlagRow>> {
        let Some(section) = self.preferred_section(command)? else {
            return Ok(Vec::new());
        };
        // An empty MATCH is an FTS5 syntax error, and a query of only short tokens
        // reduces to one. Both mean "nothing to search for", not "fail".
        let match_expr = fts_query(query);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.rank_personal
             FROM flags_fts
             JOIN flags f ON f.id = flags_fts.rowid
             JOIN commands c ON c.id = f.command_id
             WHERE flags_fts MATCH ?1 AND c.name = ?2 AND c.section = ?3
             ORDER BY bm25(flags_fts), f.source_line
             LIMIT ?4",
        )?;
        let rows = stmt
            .query_map(
                params![match_expr, command, section, limit as i64],
                row_to_flag,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn stats(&self) -> Result<(i64, i64)> {
        let commands =
            self.conn
                .query_row("SELECT count(*) FROM commands", [], |r| r.get::<_, i64>(0))?;
        let flags = self
            .conn
            .query_row("SELECT count(*) FROM flags", [], |r| r.get::<_, i64>(0))?;
        Ok((commands, flags))
    }
}

/// Turn free text into a safe FTS5 query.
///
/// User input reaches this from a shell buffer, so every term is quoted rather than passed
/// through as FTS5 syntax where `"` or `*` would be operators or a syntax error.
fn fts_query(input: &str) -> String {
    input
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .filter(|t| t.len() > 2)
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn row_to_flag(r: &rusqlite::Row) -> rusqlite::Result<FlagRow> {
    Ok(FlagRow {
        command: r.get(0)?,
        section: r.get(1)?,
        short: r.get(2)?,
        long: r.get(3)?,
        arg_type: r.get(4)?,
        arg_required: r.get::<_, i64>(5)? != 0,
        description: r.get(6)?,
        flag_group: r.get(7)?,
        source_line: r.get::<_, i64>(8)? as usize,
        rank_personal: r.get(9)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shelliq_harvest::{ParsedCommand, ParsedFlag};

    fn flag(short: &str, long: &str, desc: &str, line: usize) -> ParsedFlag {
        ParsedFlag {
            short: Some(short.to_string()),
            long: Some(long.to_string()),
            arg_type: None,
            arg_required: false,
            description: desc.to_string(),
            group: None,
            source_line: line,
        }
    }

    fn grep_fixture() -> ParsedCommand {
        ParsedCommand {
            name: "grep".into(),
            section: "1".into(),
            platform: "linux".into(),
            synopsis: "grep [OPTION...] PATTERNS [FILE...]".into(),
            description: "print lines that match patterns".into(),
            source_path: "/usr/share/man/man1/grep.1.gz".into(),
            source_hash: "deadbeef".into(),
            flags: vec![
                flag("-r", "--recursive", "Read all files under each directory", 168),
                flag(
                    "-R",
                    "--dereference-recursive",
                    "Read all files under each directory, following symlinks",
                    171,
                ),
                flag("-i", "--ignore-case", "Ignore case distinctions", 45),
            ],
        }
    }

    fn seeded() -> Index {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_command(&grep_fixture()).unwrap();
        idx
    }

    #[test]
    fn exact_lookup_is_case_sensitive() {
        let idx = seeded();
        let lower = idx.lookup_flag("grep", "-r").unwrap();
        let FlagLookup::Exact(row) = lower else {
            panic!("expected exact match for -r, got {lower:?}");
        };
        assert_eq!(row.long.as_deref(), Some("--recursive"));
    }

    /// The original complaint, as a test.
    #[test]
    fn wrong_case_is_reported_as_a_correction_not_a_match() {
        let idx = seeded();
        // Both exist for grep, so each must resolve to itself and never to the other.
        let FlagLookup::Exact(upper) = idx.lookup_flag("grep", "-R").unwrap() else {
            panic!("-R exists for grep and must match exactly");
        };
        assert_eq!(upper.long.as_deref(), Some("--dereference-recursive"));

        // -I does not exist for grep, but -i does.
        match idx.lookup_flag("grep", "-I").unwrap() {
            FlagLookup::CaseMismatch { typed, suggestion } => {
                assert_eq!(typed, "-I");
                assert_eq!(suggestion.short.as_deref(), Some("-i"));
            }
            other => panic!("expected a case-mismatch suggestion, got {other:?}"),
        }
    }

    #[test]
    fn unknown_flag_is_never_silently_accepted() {
        let idx = seeded();
        match idx.lookup_flag("grep", "--frobnicate").unwrap() {
            FlagLookup::Unknown { typed } => assert_eq!(typed, "--frobnicate"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn unknown_command_yields_unknown_rather_than_a_guess() {
        let idx = seeded();
        assert!(matches!(
            idx.lookup_flag("definitely-not-installed", "-r").unwrap(),
            FlagLookup::Unknown { .. }
        ));
    }

    #[test]
    fn flags_carry_citations() {
        let idx = seeded();
        let FlagLookup::Exact(row) = idx.lookup_flag("grep", "-r").unwrap() else {
            panic!("expected exact");
        };
        assert_eq!(row.citation(), "grep(1):168");
        assert_eq!(row.spelling(), "-r, --recursive");
    }

    #[test]
    fn description_search_finds_a_flag_by_meaning() {
        let idx = seeded();
        let hits = idx.search_flags("grep", "symlinks", 5).unwrap();
        assert!(!hits.is_empty(), "expected a description hit");
        assert_eq!(hits[0].short.as_deref(), Some("-R"));
    }

    #[test]
    fn search_input_from_a_shell_buffer_cannot_break_fts_syntax() {
        let idx = seeded();
        // Quotes and operators must be treated as text, not FTS5 syntax.
        assert!(idx.search_flags("grep", "\"unbalanced AND *", 5).is_ok());
        assert!(idx.search_flags("grep", "", 5).unwrap().is_empty());
    }

    #[test]
    fn reinserting_a_command_replaces_rather_than_duplicates() {
        let mut idx = seeded();
        idx.insert_command(&grep_fixture()).unwrap();
        let (commands, flags) = idx.stats().unwrap();
        assert_eq!(commands, 1);
        assert_eq!(flags, 3);
    }

    #[test]
    fn removed_upstream_flags_disappear_on_reharvest() {
        let mut idx = seeded();
        let mut shrunk = grep_fixture();
        shrunk.flags.truncate(1);
        idx.insert_command(&shrunk).unwrap();
        assert!(matches!(
            idx.lookup_flag("grep", "-i").unwrap(),
            FlagLookup::Unknown { .. }
        ));
    }
}
