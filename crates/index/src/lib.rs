//! The SQLite index of command facts.
//!
//! Everything shelliq states as fact — that `-r` exists, that `-R` means something else,
//! that `--block-size` takes an argument — is read from here and nowhere else. The model,
//! when there is one, never supplies a flag fact. That separation is what makes an answer
//! citable.
//!
//! Text comparison uses SQLite's default BINARY collation throughout, so `-r` and `-R` are
//! different rows and a lookup for one never returns the other.
//!
//! Every fact query is scoped to a *target*: the executable a shell would actually run for
//! a command name, resolved fresh at query time by `PATH` precedence. Two installs of the
//! same name never share facts, and a name whose live resolution was never harvested
//! answers "not indexed" rather than serving a stale or unrelated install's facts.

mod migrate;
#[cfg(test)]
mod search_relevance;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use shelliq_harvest::help_crawler::HelpNode;
use shelliq_harvest::{ParsedCommand, ParsedFlag, Target, section_rank};

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
    pub excerpt: String,
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

/// Whether a target's on-disk facts still match what is on `PATH` right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// The live executable's size and mtime match what was recorded at harvest time.
    Fresh,
    /// The live executable has changed since it was harvested; facts may no longer hold.
    PossiblyStale,
    /// This name has never been harvested under its currently resolved identity.
    Unknown,
}

/// Provenance for one citation: where the fact came from, exactly, and what it says.
#[derive(Debug, Clone)]
pub struct Provenance {
    pub command: String,
    pub section: String,
    pub source_line: usize,
    pub excerpt: String,
    pub source_path: String,
    pub source_hash: String,
    pub exec_kind: String,
    pub exec_path: Option<String>,
    pub exec_hash: Option<String>,
}

struct TargetRow {
    id: i64,
    exec_size: Option<i64>,
    exec_mtime: Option<i64>,
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
        let index = Self::init(conn)?;
        secure_permissions(path);
        Ok(index)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        if migrate::detect_version(&conn)? == 1 {
            migrate::v1_to_v2(&conn)?;
        }
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
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share"));
        base.join("shelliq/index.sqlite")
    }

    /// WAL-checkpoint and drop back to a single file, so an atomic rename leaves no
    /// orphaned `-wal`/`-shm` sidecars for the file it replaces.
    pub fn checkpoint_and_close(&self) -> Result<()> {
        self.conn.pragma_update(None, "wal_checkpoint", "TRUNCATE").ok();
        self.conn.pragma_update(None, "journal_mode", "DELETE")?;
        Ok(())
    }

    /// Every target name this index has ever harvested, for a refresh with no names given.
    pub fn all_target_names(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT DISTINCT name FROM targets ORDER BY name")?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(names)
    }

    /// Insert or replace one command and all of its flags under the given target identity.
    ///
    /// Rows are replaced wholesale rather than merged, so a flag removed upstream
    /// disappears from the index instead of lingering as a fact that is no longer true.
    pub fn insert_command(&mut self, target: &Target, cmd: &ParsedCommand) -> Result<i64> {
        let tx = self.conn.transaction()?;
        let target_id = upsert_target(&tx, target)?;

        tx.execute(
            "DELETE FROM commands WHERE target_id = ?1 AND section = ?2",
            params![target_id, cmd.section],
        )?;
        tx.execute(
            "INSERT INTO commands
                 (target_id, name, platform, section, synopsis, description,
                  source_path, source_hash, parser_version, harvested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, datetime('now'))",
            params![
                target_id,
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

        let mut flag_ids = Vec::with_capacity(cmd.flags.len());
        {
            let mut stmt = tx.prepare(
                "INSERT INTO flags
                     (command_id, short, long, arg_type, arg_required,
                      description, flag_group, source_line, excerpt)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
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
                    f.excerpt,
                ])?;
                flag_ids.push(tx.last_insert_rowid());
            }
        }

        {
            // Old edges are gone already: they cascade from the DELETE FROM commands above,
            // through flags.command_id ON DELETE CASCADE, through flag_edges' own cascade.
            let mut stmt = tx.prepare("INSERT OR IGNORE INTO flag_edges (from_flag_id, to_flag_id) VALUES (?1, ?2)")?;
            for (i, f) in cmd.flags.iter().enumerate() {
                for j in mentioned_flag_indices(&f.description, &cmd.flags, i) {
                    stmt.execute(params![flag_ids[i], flag_ids[j]])?;
                }
            }
        }

        tx.commit()?;
        Ok(command_id)
    }

    /// Insert a `--help` crawl: the root invocation's flags as the command's own, plus every
    /// subcommand path the crawl followed, each scoped under its own row in `subcommands`.
    ///
    /// Given the synthetic section `"help"`, distinct from any man page section number, so a
    /// tool with both a man page and a `--help` crawl keeps both rows: `section_rank` ranks
    /// an unrecognised section last, so `"help"` only wins `preferred_section` when no man
    /// section is competing for the name. Replaced wholesale on re-crawl, the same as
    /// `insert_command`: the old `commands` row cascades away through `subcommands` and
    /// `flags`, so a subcommand or flag dropped upstream disappears rather than lingering.
    pub fn insert_help_crawl(&mut self, target: &Target, name: &str, nodes: &[HelpNode]) -> Result<i64> {
        let root = nodes
            .iter()
            .find(|n| n.path.is_empty())
            .context("help crawl produced no root node")?;
        let cmd = ParsedCommand {
            name: name.to_string(),
            section: "help".to_string(),
            platform: shelliq_harvest::platform().to_string(),
            synopsis: String::new(),
            description: String::new(),
            source_path: target.exec_path.clone().unwrap_or_default(),
            source_hash: target.exec_hash.clone().unwrap_or_default(),
            flags: root.flags.clone(),
        };
        let command_id = self.insert_command(target, &cmd)?;

        let tx = self.conn.transaction()?;
        {
            let mut sub_stmt = tx.prepare("INSERT INTO subcommands (command_id, path, summary) VALUES (?1, ?2, '')")?;
            let mut flag_stmt = tx.prepare(
                "INSERT INTO flags
                     (command_id, subcommand_id, short, long, arg_type, arg_required,
                      description, flag_group, source_line, excerpt)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for node in nodes.iter().filter(|n| !n.path.is_empty()) {
                sub_stmt.execute(params![command_id, node.path.join(" ")])?;
                let subcommand_id = tx.last_insert_rowid();
                for f in &node.flags {
                    flag_stmt.execute(params![
                        command_id,
                        subcommand_id,
                        f.short,
                        f.long,
                        f.arg_type,
                        f.arg_required as i64,
                        f.description,
                        f.group,
                        f.source_line as i64,
                        f.excerpt,
                    ])?;
                }
            }
        }
        tx.commit()?;
        Ok(command_id)
    }

    /// The target row matching `name`'s live resolution, if it has ever been harvested, and
    /// whether that row's recorded identity still matches what is on `PATH` right now.
    fn resolved_target(&self, name: &str) -> Result<Option<(TargetRow, Freshness)>> {
        let live = shelliq_harvest::resolve_target(name, false);
        let row = self
            .conn
            .query_row(
                "SELECT id, exec_size, exec_mtime FROM targets
                 WHERE name = ?1 AND exec_kind = ?2 AND exec_path IS ?3",
                params![live.name, live.exec_kind.as_str(), live.exec_path],
                |r| {
                    Ok(TargetRow {
                        id: r.get(0)?,
                        exec_size: r.get(1)?,
                        exec_mtime: r.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(row.map(|t| {
            let fresh = match (t.exec_size, t.exec_mtime) {
                (Some(size), Some(mtime)) => live.exec_size.map(|s| s as i64) == Some(size) && live.exec_mtime == Some(mtime),
                // Builtins and absent names carry no size/mtime to compare, so their
                // identity cannot drift underneath the index the way a file's can.
                _ => true,
            };
            let freshness = if fresh { Freshness::Fresh } else { Freshness::PossiblyStale };
            (t, freshness)
        }))
    }

    /// Whether `name`'s harvested facts still match its live, resolved identity.
    pub fn freshness(&self, name: &str) -> Result<Freshness> {
        Ok(match self.resolved_target(name)? {
            Some((_, freshness)) => freshness,
            None => Freshness::Unknown,
        })
    }

    pub fn command_exists(&self, name: &str) -> Result<bool> {
        let Some((target, _)) = self.resolved_target(name)? else {
            return Ok(false);
        };
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM commands WHERE target_id = ?1",
            params![target.id],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Rank installed, indexed command names whose local documentation matches a task.
    ///
    /// Both flag prose and examples contribute. Results are checked against live `PATH`
    /// resolution through `command_exists`, so a removed or shadowed executable is never
    /// offered to the model as an available command.
    pub fn search_commands(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let terms = fts_terms(query);
        if terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let candidates = limit.saturating_mul(8).max(40);
        let mut scores = std::collections::HashMap::<String, f64>::new();
        let mut matched_terms = std::collections::HashMap::<String, std::collections::HashSet<usize>>::new();
        for (term_index, term) in terms.iter().enumerate() {
            for sql in [
                "SELECT c.name FROM examples_fts JOIN examples e ON e.id = examples_fts.rowid JOIN commands c ON c.id = e.command_id WHERE examples_fts MATCH ?1 ORDER BY bm25(examples_fts) LIMIT ?2",
                "SELECT c.name FROM flags_fts JOIN flags f ON f.id = flags_fts.rowid JOIN commands c ON c.id = f.command_id WHERE flags_fts MATCH ?1 ORDER BY bm25(flags_fts) LIMIT ?2",
            ] {
                let mut stmt = self.conn.prepare(sql)?;
                let names = stmt
                    .query_map(params![term, candidates as i64], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                let mut seen = std::collections::HashSet::new();
                for (rank, name) in names.into_iter().filter(|name| seen.insert(name.clone())).enumerate() {
                    *scores.entry(name.clone()).or_default() += 1.0 / (60.0 + rank as f64 + 1.0);
                    matched_terms.entry(name).or_default().insert(term_index);
                }
            }
        }
        let mut names: Vec<_> = scores.into_iter().collect();
        names.sort_by(|(left_name, left_score), (right_name, right_score)| {
            right_score.total_cmp(left_score).then_with(|| left_name.cmp(right_name))
        });

        let mut available = Vec::new();
        for (name, _) in names {
            let required_coverage = terms.len().min(2);
            if matched_terms.get(&name).map_or(0, std::collections::HashSet::len) >= required_coverage
                && self.command_exists(&name)?
            {
                available.push(name);
                if available.len() == limit {
                    break;
                }
            }
        }
        Ok(available)
    }

    /// The section that answers a bare command lookup, by `SECTION_PREFERENCE`.
    fn preferred_section(&self, command: &str) -> Result<Option<(i64, String)>> {
        let Some((target, _)) = self.resolved_target(command)? else {
            return Ok(None);
        };
        let mut stmt = self.conn.prepare("SELECT section FROM commands WHERE target_id = ?1")?;
        let mut sections = stmt
            .query_map(params![target.id], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        sections.sort_by_key(|s| section_rank(s));
        Ok(sections.into_iter().next().map(|s| (target.id, s)))
    }

    /// All flags for a command, preferring the section a shell user means.
    ///
    /// Scoped to flags with no subcommand: a subcommand's own flags are reached through its
    /// own path, never folded into the command's global list.
    pub fn flags_for(&self, command: &str) -> Result<Vec<FlagRow>> {
        let Some((target_id, section)) = self.preferred_section(command)? else {
            return Ok(Vec::new());
        };
        let mut stmt = self.conn.prepare(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.target_id = ?1 AND c.section = ?2 AND f.subcommand_id IS NULL
             ORDER BY f.rank_personal DESC, f.source_line",
        )?;
        let rows = stmt
            .query_map(params![target_id, section], row_to_flag)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Check one flag token, case-sensitively, then case-insensitively as a suggestion.
    pub fn lookup_flag(&self, command: &str, token: &str) -> Result<FlagLookup> {
        let Some((target_id, section)) = self.preferred_section(command)? else {
            return Ok(FlagLookup::Unknown {
                typed: token.to_string(),
            });
        };

        let exact = self.query_one(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.target_id = ?1 AND c.section = ?2 AND f.subcommand_id IS NULL
               AND (f.short = ?3 OR f.long = ?3)
             LIMIT 1",
            target_id,
            &section,
            token,
        )?;
        if let Some(row) = exact {
            return Ok(FlagLookup::Exact(Box::new(row)));
        }

        // `lower()` is ASCII-only in SQLite, which is correct here: flag names are ASCII.
        let folded = self.query_one(
            "SELECT c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM flags f JOIN commands c ON c.id = f.command_id
             WHERE c.target_id = ?1 AND c.section = ?2 AND f.subcommand_id IS NULL
               AND (lower(f.short) = lower(?3) OR lower(f.long) = lower(?3))
             LIMIT 1",
            target_id,
            &section,
            token,
        )?;
        Ok(match folded {
            Some(row) => FlagLookup::CaseMismatch {
                typed: token.to_string(),
                suggestion: Box::new(row),
            },
            None => FlagLookup::Unknown {
                typed: token.to_string(),
            },
        })
    }

    fn query_one(&self, sql: &str, target_id: i64, section: &str, token: &str) -> Result<Option<FlagRow>> {
        let mut stmt = self.conn.prepare(sql)?;
        let row = stmt.query_row(params![target_id, section, token], row_to_flag).optional()?;
        Ok(row)
    }

    /// Full-text search over flag descriptions, fused with a search over tldr examples and a
    /// one-hop expansion through description cross-references.
    ///
    /// Descriptions speak mechanism ("moved to a different location"); tldr examples speak
    /// task ("follow redirect"). A query in task language can miss the description search
    /// entirely, so the lists are merged by reciprocal rank rather than trusting any one
    /// alone — this is the path that turns "follow redirect" into `-L, --location`.
    pub fn search_flags(&self, command: &str, query: &str, limit: usize) -> Result<Vec<FlagRow>> {
        let Some((target_id, section)) = self.preferred_section(command)? else {
            return Ok(Vec::new());
        };
        // An empty MATCH is an FTS5 syntax error, and a query of only short tokens
        // reduces to one. Both mean "nothing to search for", not "fail".
        let match_expr = fts_query(query);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let candidates = limit.saturating_mul(4).max(20);
        let by_description = self.search_flags_fts(target_id, &section, &match_expr, candidates)?;
        let by_example = self.search_flags_via_examples(target_id, &section, &match_expr, candidates)?;
        let by_edges = self.search_flags_via_edges(&by_description, target_id, &section)?;
        Ok(rrf_merge(
            &[(&by_description, 1.0), (&by_example, 1.0), (&by_edges, EDGE_EXPANSION_WEIGHT)],
            limit,
        ))
    }

    /// Description-only search, with no tldr or cross-reference fusion — the "before" side
    /// of the `search_relevance` Recall@5/MRR measurement. Never called outside tests.
    #[cfg(test)]
    fn search_flags_description_only(&self, command: &str, query: &str, limit: usize) -> Result<Vec<FlagRow>> {
        let Some((target_id, section)) = self.preferred_section(command)? else {
            return Ok(Vec::new());
        };
        let match_expr = fts_query(query);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.search_flags_fts(target_id, &section, &match_expr, limit)?;
        Ok(rows.into_iter().map(|(_, f)| f).collect())
    }

    fn search_flags_fts(
        &self,
        target_id: i64,
        section: &str,
        match_expr: &str,
        candidates: usize,
    ) -> Result<Vec<(i64, FlagRow)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM flags_fts
             JOIN flags f ON f.id = flags_fts.rowid
             JOIN commands c ON c.id = f.command_id
             WHERE flags_fts MATCH ?1 AND c.target_id = ?2 AND c.section = ?3
               AND f.subcommand_id IS NULL
             ORDER BY bm25(flags_fts), f.source_line
             LIMIT ?4",
        )?;
        let rows = stmt
            .query_map(
                params![match_expr, target_id, section, candidates as i64],
                row_to_flag_with_id,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Flags reached through tldr examples: an example matches the query, and a flag it
    /// mentions was confirmed (at ingestion time) to exist on this target — see
    /// `insert_tldr_examples`. A flag mentioned by several matching examples keeps only its
    /// best-ranked (first-seen) appearance; `bm25()` can only be ordered directly by FTS5,
    /// not wrapped in an aggregate under `GROUP BY`, so deduplication happens here instead.
    fn search_flags_via_examples(
        &self,
        target_id: i64,
        section: &str,
        match_expr: &str,
        candidates: usize,
    ) -> Result<Vec<(i64, FlagRow)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM examples_fts
             JOIN examples e ON e.id = examples_fts.rowid
             JOIN example_flags ef ON ef.example_id = e.id
             JOIN flags f ON f.id = ef.flag_id
             JOIN commands c ON c.id = f.command_id
             WHERE examples_fts MATCH ?1 AND c.target_id = ?2 AND c.section = ?3
               AND f.subcommand_id IS NULL
             ORDER BY bm25(examples_fts), f.source_line
             LIMIT ?4",
        )?;
        let rows = stmt
            .query_map(
                params![match_expr, target_id, section, candidates as i64],
                row_to_flag_with_id,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = std::collections::HashSet::new();
        Ok(rows.into_iter().filter(|(id, _)| seen.insert(*id)).collect())
    }

    /// One hop through `flag_edges` from each flag `matches` found by description search:
    /// curl's `--location-trusted` description says "Like -L, --location, but…", so a query
    /// that only matches `--location-trusted`'s own description should still surface `-L`.
    /// Never expands past one hop — `search_flags` weights this list down relative to a
    /// direct match, per PLAN.md's "score penalty" design.
    fn search_flags_via_edges(&self, matches: &[(i64, FlagRow)], target_id: i64, section: &str) -> Result<Vec<(i64, FlagRow)>> {
        if matches.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT f.id, c.name, c.section, f.short, f.long, f.arg_type, f.arg_required,
                    f.description, f.flag_group, f.source_line, f.excerpt, f.rank_personal
             FROM flag_edges fe
             JOIN flags f ON f.id = fe.to_flag_id
             JOIN commands c ON c.id = f.command_id
             WHERE fe.from_flag_id = ?1 AND c.target_id = ?2 AND c.section = ?3
               AND f.subcommand_id IS NULL",
        )?;
        let mut seen = std::collections::HashSet::new();
        let mut expanded = Vec::new();
        for (from_id, _) in matches {
            let rows = stmt
                .query_map(params![from_id, target_id, section], row_to_flag_with_id)?
                .collect::<Result<Vec<_>, _>>()?;
            for (id, row) in rows {
                if seen.insert(id) {
                    expanded.push((id, row));
                }
            }
        }
        Ok(expanded)
    }

    /// Ingest tldr examples for `command`, validating every mentioned flag against flags
    /// already indexed for this target before it can boost anything. tldr pages are
    /// generic; a flag they mention that this target does not actually have must never
    /// become a fact, so unmatched spellings are silently dropped rather than inserted.
    ///
    /// Rows are replaced wholesale like `insert_command`: previously ingested tldr examples
    /// for this command are deleted before the fresh set is inserted.
    pub fn insert_tldr_examples(&mut self, command: &str) -> Result<usize> {
        let Some((target_id, section)) = self.preferred_section(command)? else {
            return Ok(0);
        };
        let (command_id, name, platform): (i64, String, String) = self.conn.query_row(
            "SELECT id, name, platform FROM commands WHERE target_id = ?1 AND section = ?2",
            params![target_id, section],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;

        let pages = shelliq_harvest::tldr::harvest_tldr(&name, &platform)?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM examples WHERE command_id = ?1 AND source = 'tldr'",
            params![command_id],
        )?;

        let mut inserted = 0;
        for example in &pages {
            tx.execute(
                "INSERT INTO examples (command_id, text, description, source) VALUES (?1, ?2, ?3, 'tldr')",
                params![command_id, example.text, example.description],
            )?;
            let example_id = tx.last_insert_rowid();
            inserted += 1;

            for spelling in &example.flags {
                let flag_id: Option<i64> = tx
                    .query_row(
                        "SELECT id FROM flags WHERE command_id = ?1 AND subcommand_id IS NULL
                         AND (short = ?2 OR long = ?2)",
                        params![command_id, spelling],
                        |r| r.get(0),
                    )
                    .optional()?;
                let Some(flag_id) = flag_id else { continue };
                tx.execute(
                    "INSERT OR IGNORE INTO example_flags (example_id, flag_id) VALUES (?1, ?2)",
                    params![example_id, flag_id],
                )?;
                tx.execute("UPDATE flags SET rank_tldr = rank_tldr + 1 WHERE id = ?1", params![flag_id])?;
            }
        }

        tx.commit()?;
        Ok(inserted)
    }

    /// Provenance for one citation: which target, source file, and exact excerpt it came
    /// from, for `shelliq source` to quote verbatim rather than merely re-describe.
    pub fn provenance(&self, command: &str, section: &str, source_line: usize) -> Result<Option<Provenance>> {
        let Some((target, _)) = self.resolved_target(command)? else {
            return Ok(None);
        };
        self.conn
            .query_row(
                "SELECT c.name, c.section, f.source_line, f.excerpt, c.source_path, c.source_hash,
                        t.exec_kind, t.exec_path, t.exec_hash
                 FROM flags f
                 JOIN commands c ON c.id = f.command_id
                 JOIN targets t ON t.id = c.target_id
                 WHERE c.target_id = ?1 AND c.section = ?2 AND f.source_line = ?3
                 LIMIT 1",
                params![target.id, section, source_line as i64],
                |r| {
                    Ok(Provenance {
                        command: r.get(0)?,
                        section: r.get(1)?,
                        source_line: r.get::<_, i64>(2)? as usize,
                        excerpt: r.get(3)?,
                        source_path: r.get(4)?,
                        source_hash: r.get(5)?,
                        exec_kind: r.get(6)?,
                        exec_path: r.get(7)?,
                        exec_hash: r.get(8)?,
                    })
                },
            )
            .optional()
            .context("querying provenance")
    }

    pub fn stats(&self) -> Result<(i64, i64)> {
        let commands = self
            .conn
            .query_row("SELECT count(*) FROM commands", [], |r| r.get::<_, i64>(0))?;
        let flags = self
            .conn
            .query_row("SELECT count(*) FROM flags", [], |r| r.get::<_, i64>(0))?;
        Ok((commands, flags))
    }
}

/// Find or create the stored target row matching this identity, refreshing its recorded
/// size/hash/mtime in place.
///
/// Identity is `(name, exec_kind, exec_path)`. A different `exec_path` — a second install —
/// is a different row; the same path with new content updates the existing row's recorded
/// state instead, since it is the same install, just changed.
fn upsert_target(tx: &rusqlite::Transaction, target: &Target) -> Result<i64> {
    let existing: Option<i64> = tx
        .query_row(
            "SELECT id FROM targets WHERE name = ?1 AND exec_kind = ?2 AND exec_path IS ?3",
            params![target.name, target.exec_kind.as_str(), target.exec_path],
            |r| r.get(0),
        )
        .optional()?;

    if let Some(id) = existing {
        tx.execute(
            "UPDATE targets SET exec_hash = ?1, exec_size = ?2, exec_mtime = ?3, last_checked = datetime('now')
             WHERE id = ?4",
            params![target.exec_hash, target.exec_size.map(|s| s as i64), target.exec_mtime, id,],
        )?;
        Ok(id)
    } else {
        tx.execute(
            "INSERT INTO targets
                 (name, platform, exec_kind, exec_path, exec_hash, exec_size, exec_mtime,
                  first_seen, last_checked)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'), datetime('now'))",
            params![
                target.name,
                target.platform,
                target.exec_kind.as_str(),
                target.exec_path,
                target.exec_hash,
                target.exec_size.map(|s| s as i64),
                target.exec_mtime,
            ],
        )?;
        Ok(tx.last_insert_rowid())
    }
}

/// Restrict an index file and its WAL/SHM sidecars to owner-only.
///
/// The index is a plain SQLite file with no encryption of its own, so its permissions are
/// the only thing standing between "local machine" and "any user on this machine can read
/// every man page fact and file path ever harvested".
pub fn secure_permissions(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    for suffix in ["", "-wal", "-shm"] {
        let candidate = format!("{}{suffix}", path.display());
        if let Ok(meta) = std::fs::metadata(&candidate) {
            let mut perms = meta.permissions();
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(&candidate, perms);
        }
    }
}

/// Turn free text into a safe FTS5 query.
///
/// User input reaches this from a shell buffer, so every term is quoted rather than passed
/// through as FTS5 syntax where `"` or `*` would be operators or a syntax error.
fn fts_query(input: &str) -> String {
    fts_terms(input).join(" OR ")
}

fn fts_terms(input: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "and", "for", "from", "into", "the", "this", "that", "under", "using", "while", "with",
    ];
    input
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|term| term.len() >= 3 && !STOP_WORDS.contains(&term.as_str()))
        .map(|term| format!("\"{}\"", term.replace('"', "")))
        .collect()
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
        excerpt: r.get(9)?,
        rank_personal: r.get(10)?,
    })
}

fn row_to_flag_with_id(r: &rusqlite::Row) -> rusqlite::Result<(i64, FlagRow)> {
    Ok((
        r.get(0)?,
        FlagRow {
            command: r.get(1)?,
            section: r.get(2)?,
            short: r.get(3)?,
            long: r.get(4)?,
            arg_type: r.get(5)?,
            arg_required: r.get::<_, i64>(6)? != 0,
            description: r.get(7)?,
            flag_group: r.get(8)?,
            source_line: r.get::<_, i64>(9)? as usize,
            excerpt: r.get(10)?,
            rank_personal: r.get(11)?,
        },
    ))
}

/// Reciprocal rank fusion: merges ranked lists by `weight / (k + rank)`, so a flag that
/// ranks well in one list outranks one that ranks moderately in several, without any list's
/// raw score scale (BM25 here) needing to be comparable to another's. A list's weight below
/// 1.0 is PLAN.md's "score penalty" — currently used to discount one-hop edge expansion
/// relative to a direct match.
const RRF_K: f64 = 60.0;

/// Edge-expanded flags are reached indirectly (a neighbour's description mentioned them, not
/// the query itself), so they are weighted down relative to a direct description or example
/// match — see `search_flags_via_edges` and PLAN.md section 4, remedy 2.
const EDGE_EXPANSION_WEIGHT: f64 = 0.5;

fn rrf_merge(lists: &[(&[(i64, FlagRow)], f64)], limit: usize) -> Vec<FlagRow> {
    let mut scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    let mut rows: std::collections::HashMap<i64, FlagRow> = std::collections::HashMap::new();
    for (list, weight) in lists {
        for (rank, (id, row)) in list.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += weight / (RRF_K + rank as f64 + 1.0);
            rows.entry(*id).or_insert_with(|| row.clone());
        }
    }
    let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
    ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then_with(|| x.0.cmp(&y.0)));
    ranked
        .into_iter()
        .take(limit)
        .filter_map(|(id, _)| rows.remove(&id))
        .collect()
}

/// Indices into `flags` whose short/long spelling appears as a standalone token inside
/// `description`, excluding `flags[exclude]` itself. Token boundaries are whitespace and
/// trailing punctuation, so "moved to a different location" never matches `-L` or
/// `--location` — only an exact spelling like "-L, --location" does. This is deliberately
/// stricter than a substring search: prose mentions a flag for many reasons besides being
/// related to it, and a false edge here silently distorts search for every future query.
fn mentioned_flag_indices(description: &str, flags: &[ParsedFlag], exclude: usize) -> Vec<usize> {
    let tokens: std::collections::HashSet<&str> = description
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '(' | ')')))
        .collect();
    flags
        .iter()
        .enumerate()
        .filter(|(j, other)| {
            *j != exclude
                && (other.short.as_deref().is_some_and(|s| tokens.contains(s))
                    || other.long.as_deref().is_some_and(|l| tokens.contains(l)))
        })
        .map(|(j, _)| j)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shelliq_harvest::{ExecKind, ParsedCommand, ParsedFlag};

    fn flag(short: &str, long: &str, desc: &str, line: usize) -> ParsedFlag {
        ParsedFlag {
            short: Some(short.to_string()),
            long: Some(long.to_string()),
            arg_type: None,
            arg_required: false,
            description: desc.to_string(),
            group: None,
            source_line: line,
            excerpt: format!("     {short}, {long}  {desc}"),
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

    fn grep_target() -> Target {
        shelliq_harvest::resolve_target("grep", false)
    }

    fn seeded() -> Index {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_command(&grep_target(), &grep_fixture()).unwrap();
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
    fn command_search_ranks_an_installed_command_by_its_flag_prose() {
        let idx = seeded();
        let hits = idx.search_commands("symlinks", 6).unwrap();
        assert_eq!(hits.first().map(String::as_str), Some("grep"));
    }

    #[test]
    fn command_search_requires_two_matching_terms_for_multi_term_tasks() {
        let idx = seeded();
        assert!(idx.search_commands("symlinks transcoding", 6).unwrap().is_empty());
    }

    #[test]
    fn fts_terms_drop_short_and_common_instruction_words_before_quoting() {
        assert_eq!(
            fts_terms("a file to copy with metadata"),
            ["\"file\"", "\"copy\"", "\"metadata\""]
        );
    }

    #[test]
    fn reinserting_a_command_replaces_rather_than_duplicates() {
        let mut idx = seeded();
        idx.insert_command(&grep_target(), &grep_fixture()).unwrap();
        let (commands, flags) = idx.stats().unwrap();
        assert_eq!(commands, 1);
        assert_eq!(flags, 3);
    }

    #[test]
    fn removed_upstream_flags_disappear_on_reharvest() {
        let mut idx = seeded();
        let mut shrunk = grep_fixture();
        shrunk.flags.truncate(1);
        idx.insert_command(&grep_target(), &shrunk).unwrap();
        assert!(matches!(idx.lookup_flag("grep", "-i").unwrap(), FlagLookup::Unknown { .. }));
    }

    #[test]
    fn distinct_installs_of_the_same_name_get_distinct_targets() {
        let mut idx = Index::open_in_memory().unwrap();
        let a = Target {
            name: "widget".into(),
            platform: "linux".into(),
            exec_kind: ExecKind::File,
            exec_path: Some("/usr/bin/widget".into()),
            exec_hash: Some("aaa".into()),
            exec_size: Some(100),
            exec_mtime: Some(1000),
        };
        let b = Target {
            name: "widget".into(),
            platform: "linux".into(),
            exec_kind: ExecKind::File,
            exec_path: Some("/opt/homebrew/bin/widget".into()),
            exec_hash: Some("bbb".into()),
            exec_size: Some(200),
            exec_mtime: Some(2000),
        };
        let mut cmd = grep_fixture();
        cmd.name = "widget".into();

        idx.insert_command(&a, &cmd).unwrap();
        idx.insert_command(&b, &cmd).unwrap();

        let target_count: i64 = idx
            .conn
            .query_row("SELECT count(*) FROM targets WHERE name = 'widget'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(target_count, 2, "two different installs must not collapse into one target");

        let distinct_target_ids: i64 = idx
            .conn
            .query_row(
                "SELECT count(DISTINCT target_id) FROM commands WHERE name = 'widget'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(distinct_target_ids, 2);
    }

    #[test]
    fn reinserting_the_same_install_updates_rather_than_duplicates_the_target() {
        let mut idx = Index::open_in_memory().unwrap();
        let mut cmd = grep_fixture();
        cmd.name = "widget".into();
        let v1 = Target {
            name: "widget".into(),
            platform: "linux".into(),
            exec_kind: ExecKind::File,
            exec_path: Some("/usr/bin/widget".into()),
            exec_hash: Some("aaa".into()),
            exec_size: Some(100),
            exec_mtime: Some(1000),
        };
        idx.insert_command(&v1, &cmd).unwrap();

        // Same path, upgraded content: still the same install, not a new one.
        let v2 = Target {
            exec_hash: Some("ccc".into()),
            exec_size: Some(150),
            exec_mtime: Some(3000),
            ..v1.clone()
        };
        idx.insert_command(&v2, &cmd).unwrap();

        let target_count: i64 = idx
            .conn
            .query_row("SELECT count(*) FROM targets WHERE name = 'widget'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(target_count, 1);
        let hash: String = idx
            .conn
            .query_row("SELECT exec_hash FROM targets WHERE name = 'widget'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(hash, "ccc");
    }

    #[test]
    fn flags_scoped_to_a_subcommand_do_not_leak_into_the_global_lookup() {
        let idx = seeded();
        let command_id: i64 = idx
            .conn
            .query_row("SELECT id FROM commands WHERE name = 'grep'", [], |r| r.get(0))
            .unwrap();
        idx.conn
            .execute(
                "INSERT INTO subcommands (command_id, path, summary) VALUES (?1, 'grep sub', '')",
                [command_id],
            )
            .unwrap();
        let subcommand_id = idx.conn.last_insert_rowid();
        idx.conn
            .execute(
                "INSERT INTO flags (command_id, subcommand_id, short, long, description, source_line)
                 VALUES (?1, ?2, '-z', '--zonked', 'only under the subcommand', 999)",
                params![command_id, subcommand_id],
            )
            .unwrap();

        assert!(matches!(idx.lookup_flag("grep", "-z").unwrap(), FlagLookup::Unknown { .. }));
        assert!(
            idx.flags_for("grep")
                .unwrap()
                .iter()
                .all(|f| f.short.as_deref() != Some("-z"))
        );
    }

    fn help_crawl_nodes() -> Vec<shelliq_harvest::help_crawler::HelpNode> {
        vec![
            shelliq_harvest::help_crawler::HelpNode {
                path: vec![],
                flags: vec![flag("-h", "--help", "Show help", 1)],
                subcommands: vec!["list".into()],
            },
            shelliq_harvest::help_crawler::HelpNode {
                path: vec!["list".into()],
                flags: vec![flag("-a", "--all", "List everything", 1)],
                subcommands: vec![],
            },
        ]
    }

    #[test]
    fn help_crawl_flags_are_scoped_to_their_subcommand_path() {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_help_crawl(&grep_target(), "grep", &help_crawl_nodes()).unwrap();

        assert!(matches!(idx.lookup_flag("grep", "-h").unwrap(), FlagLookup::Exact(_)));
        assert!(matches!(idx.lookup_flag("grep", "-a").unwrap(), FlagLookup::Unknown { .. }));

        let path: String = idx.conn.query_row("SELECT path FROM subcommands", [], |r| r.get(0)).unwrap();
        assert_eq!(path, "list");
    }

    #[test]
    fn reinserting_a_help_crawl_replaces_subcommands_rather_than_duplicating() {
        let mut idx = Index::open_in_memory().unwrap();
        idx.insert_help_crawl(&grep_target(), "grep", &help_crawl_nodes()).unwrap();
        idx.insert_help_crawl(&grep_target(), "grep", &help_crawl_nodes()).unwrap();

        let subcommands: i64 = idx
            .conn
            .query_row("SELECT count(*) FROM subcommands", [], |r| r.get(0))
            .unwrap();
        assert_eq!(subcommands, 1);
        let scoped_flags: i64 = idx
            .conn
            .query_row("SELECT count(*) FROM flags WHERE subcommand_id IS NOT NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(scoped_flags, 1);
    }

    #[test]
    fn freshness_is_unknown_before_the_first_harvest() {
        let idx = Index::open_in_memory().unwrap();
        assert_eq!(idx.freshness("grep").unwrap(), Freshness::Unknown);
    }

    #[test]
    fn freshness_is_fresh_immediately_after_a_harvest() {
        let idx = seeded();
        assert_eq!(idx.freshness("grep").unwrap(), Freshness::Fresh);
    }

    #[test]
    fn freshness_is_possibly_stale_when_the_recorded_identity_no_longer_matches() {
        let idx = seeded();
        idx.conn
            .execute("UPDATE targets SET exec_size = exec_size + 1 WHERE name = 'grep'", [])
            .unwrap();
        assert_eq!(idx.freshness("grep").unwrap(), Freshness::PossiblyStale);
    }

    #[test]
    fn provenance_carries_the_verbatim_excerpt_and_source_identity() {
        let idx = seeded();
        let prov = idx.provenance("grep", "1", 168).unwrap().expect("citation should exist");
        assert_eq!(prov.command, "grep");
        assert_eq!(prov.source_path, "/usr/share/man/man1/grep.1.gz");
        assert_eq!(prov.source_hash, "deadbeef");
        assert!(prov.excerpt.contains("--recursive"));
    }

    fn curl_fixture() -> ParsedCommand {
        ParsedCommand {
            name: "curl".into(),
            section: "1".into(),
            platform: "linux".into(),
            synopsis: "curl [options] [URL...]".into(),
            description: "transfer a URL".into(),
            source_path: "/usr/share/man/man1/curl.1.gz".into(),
            source_hash: "deadbeef".into(),
            flags: vec![
                flag("-L", "--location", "If the server reports that the requested page has moved to a different location", 200),
                flag("-D", "--dump-header", "Write the protocol headers to the specified file", 210),
                ParsedFlag {
                    short: None,
                    long: Some("--location-trusted".to_string()),
                    arg_type: None,
                    arg_required: false,
                    description: "Like -L, --location, but will also send the user name and password to all hosts that the site may send you onward to".to_string(),
                    group: None,
                    source_line: 220,
                    excerpt: "     --location-trusted  Like -L, --location, but...".to_string(),
                },
            ],
        }
    }

    /// The acceptance criterion in PLAN.md, as a test: description search alone cannot find
    /// `-L, --location` from "follow redirect" (the phrase never appears in curl's man page
    /// description), but tldr ingestion bridges task language to the flag.
    #[test]
    fn tldr_ingestion_finds_a_flag_by_task_language_a_description_search_misses() {
        let mut idx = Index::open_in_memory().unwrap();
        let target = shelliq_harvest::resolve_target("curl", false);
        idx.insert_command(&target, &curl_fixture()).unwrap();

        assert!(
            idx.search_flags("curl", "follow redirect", 5).unwrap().is_empty(),
            "description search should not yet find -L for task language"
        );

        let inserted = idx.insert_tldr_examples("curl").unwrap();
        assert!(inserted > 0, "expected curl's vendored tldr page to yield examples");

        let hits = idx.search_flags("curl", "follow redirect", 5).unwrap();
        assert!(
            hits.iter().any(|f| f.short.as_deref() == Some("-L")),
            "expected -L to be found via tldr examples, got {hits:?}"
        );
    }

    #[test]
    fn tldr_examples_never_credit_a_flag_the_target_does_not_have() {
        let mut idx = Index::open_in_memory().unwrap();
        let target = shelliq_harvest::resolve_target("curl", false);
        let mut cmd = curl_fixture();
        // Drop -L so it is not one of this target's actual flags.
        cmd.flags.truncate(1);
        cmd.flags[0] = flag("-D", "--dump-header", "Write the protocol headers to the specified file", 210);
        idx.insert_command(&target, &cmd).unwrap();

        idx.insert_tldr_examples("curl").unwrap();

        let example_flag_count: i64 = idx
            .conn
            .query_row(
                "SELECT count(*) FROM example_flags ef
                 JOIN flags f ON f.id = ef.flag_id
                 WHERE f.short = '-L'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            example_flag_count, 0,
            "-L must not be credited when it is not one of this target's flags"
        );
    }

    /// PLAN.md's own motivating example: curl's `--location-trusted` description reads "Like
    /// -L, --location, but...", and edge extraction must turn that into a `flag_edges` row.
    #[test]
    fn cross_reference_edge_extracted_from_flag_description() {
        let idx = seeded_curl();
        let edge_count: i64 = idx
            .conn
            .query_row(
                "SELECT count(*) FROM flag_edges fe
                 JOIN flags a ON a.id = fe.from_flag_id
                 JOIN flags b ON b.id = fe.to_flag_id
                 WHERE a.long = '--location-trusted' AND b.short = '-L'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(edge_count, 1, "expected --location-trusted's description to link to -L");
    }

    /// PLAN.md's own counter-example: "moved to a different location" is prose, not a mention
    /// of `-L`/`--location` — the token "location" never matches the exact spelling
    /// "--location", so no edge should be created from it.
    #[test]
    fn vague_prose_does_not_create_a_spurious_edge() {
        let idx = seeded_curl();
        let edge_count: i64 = idx
            .conn
            .query_row(
                "SELECT count(*) FROM flag_edges fe
                 JOIN flags a ON a.id = fe.from_flag_id
                 JOIN flags b ON b.id = fe.to_flag_id
                 WHERE a.short = '-L' AND b.short = '-L'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            edge_count, 0,
            "\"moved to a different location\" must not create a self-edge for -L"
        );
    }

    /// The end-to-end payoff: a query that only matches `--location-trusted`'s description
    /// should still surface `-L` through one-hop edge expansion.
    #[test]
    fn search_expands_one_hop_through_cross_referenced_flags() {
        let idx = seeded_curl();
        let hits = idx.search_flags("curl", "user name and password to all hosts", 5).unwrap();
        assert!(
            hits.iter().any(|f| f.short.as_deref() == Some("-L")),
            "expected -L to be surfaced via one-hop expansion from --location-trusted, got {hits:?}"
        );
    }

    #[test]
    fn reingesting_tldr_examples_replaces_rather_than_duplicates() {
        let mut idx = seeded_curl();
        idx.insert_tldr_examples("curl").unwrap();
        let (first_count, first_links): (i64, i64) = idx
            .conn
            .query_row(
                "SELECT (SELECT count(*) FROM examples WHERE source = 'tldr'),
                        (SELECT count(*) FROM example_flags)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();

        idx.insert_tldr_examples("curl").unwrap();
        let (second_count, second_links): (i64, i64) = idx
            .conn
            .query_row(
                "SELECT (SELECT count(*) FROM examples WHERE source = 'tldr'),
                        (SELECT count(*) FROM example_flags)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();

        assert_eq!(first_count, second_count);
        assert_eq!(first_links, second_links);
    }

    fn seeded_curl() -> Index {
        let mut idx = Index::open_in_memory().unwrap();
        let target = shelliq_harvest::resolve_target("curl", false);
        idx.insert_command(&target, &curl_fixture()).unwrap();
        idx
    }

    #[test]
    fn migration_is_exercised_end_to_end_through_index_open() {
        let dir = tempfile();
        let conn = Connection::open(&dir).unwrap();
        conn.execute_batch(include_str!("schema_v1_fixture.sql")).unwrap();
        conn.execute(
            "INSERT INTO commands (name, platform, section, synopsis, description,
                                    source_path, source_hash, parser_version, harvested_at)
             VALUES ('grep', 'linux', '1', '', '', '', '', 1, datetime('now'))",
            [],
        )
        .unwrap();
        drop(conn);

        let idx = Index::open(&dir).unwrap();
        assert!(idx.command_exists("grep").is_ok());
        let (commands, _) = idx.stats().unwrap();
        assert_eq!(commands, 1);
        std::fs::remove_file(&dir).ok();
    }

    fn tempfile() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shelliq-migrate-test-{}.sqlite", std::process::id()))
    }
}
