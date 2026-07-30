//! Schema migration from the P0 (v1) database to the current (v2) shape.
//!
//! v1 had no notion of executable identity: `commands` was keyed on `(name, platform,
//! section)` alone, so a fact harvested from one install of a name could be served for a
//! different install with nothing to tell them apart. v2 adds `targets` and keys `commands`
//! on `(target_id, section)` instead.
//!
//! A v1 database predates the concept, so migration backfills one synthetic target per
//! distinct `(name, platform)` already on file, with `exec_kind = 'file'` and everything
//! else unknown. Nothing is dropped; the first live lookup after migration re-establishes a
//! real identity through the normal insert path.

use anyhow::{Context, Result};
use rusqlite::Connection;

pub fn v1_to_v2(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "foreign_keys", "OFF")?;
    conn.execute_batch(
        "BEGIN;

         ALTER TABLE flags ADD COLUMN excerpt TEXT NOT NULL DEFAULT '';

         CREATE TABLE targets (
             id           INTEGER PRIMARY KEY,
             name         TEXT NOT NULL,
             platform     TEXT NOT NULL,
             exec_kind    TEXT NOT NULL,
             exec_path    TEXT,
             exec_hash    TEXT,
             exec_size    INTEGER,
             exec_mtime   INTEGER,
             first_seen   TEXT NOT NULL,
             last_checked TEXT NOT NULL
         );
         CREATE INDEX targets_identity ON targets (name, exec_kind, exec_path);

         INSERT INTO targets (name, platform, exec_kind, exec_path, exec_hash, exec_size, exec_mtime, first_seen, last_checked)
         SELECT DISTINCT name, platform, 'file', NULL, NULL, NULL, NULL, datetime('now'), datetime('now')
         FROM commands;

         CREATE TABLE commands_v2 (
             id             INTEGER PRIMARY KEY,
             target_id      INTEGER NOT NULL REFERENCES targets (id) ON DELETE CASCADE,
             name           TEXT NOT NULL,
             platform       TEXT NOT NULL,
             section        TEXT NOT NULL,
             version        TEXT,
             synopsis       TEXT NOT NULL DEFAULT '',
             description    TEXT NOT NULL DEFAULT '',
             source_path    TEXT NOT NULL DEFAULT '',
             source_hash    TEXT NOT NULL DEFAULT '',
             parser_version INTEGER NOT NULL DEFAULT 0,
             harvested_at   TEXT NOT NULL,
             UNIQUE (target_id, section)
         );

         INSERT INTO commands_v2
             SELECT c.id,
                    (SELECT t.id FROM targets t WHERE t.name = c.name AND t.platform = c.platform),
                    c.name, c.platform, c.section, c.version, c.synopsis, c.description,
                    c.source_path, c.source_hash, c.parser_version, c.harvested_at
             FROM commands c;

         DROP TABLE commands;
         ALTER TABLE commands_v2 RENAME TO commands;

         INSERT INTO meta (key, value) VALUES ('schema_version', '2')
             ON CONFLICT (key) DO UPDATE SET value = '2';

         COMMIT;",
    )
    .context("migrating schema v1 to v2")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// `0` for a brand-new database, `1` for the pre-`targets` P0 shape, `2` for current.
pub fn detect_version(conn: &Connection) -> Result<i64> {
    let has_commands: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'commands'",
        [],
        |r| r.get(0),
    )?;
    if has_commands == 0 {
        return Ok(0);
    }
    let has_target_id: i64 = conn.query_row(
        "SELECT count(*) FROM pragma_table_info('commands') WHERE name = 'target_id'",
        [],
        |r| r.get(0),
    )?;
    Ok(if has_target_id > 0 { 2 } else { 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1_FIXTURE: &str = include_str!("schema_v1_fixture.sql");

    fn v1_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(V1_FIXTURE).unwrap();
        conn.execute(
            "INSERT INTO commands (name, platform, section, synopsis, description,
                                    source_path, source_hash, parser_version, harvested_at)
             VALUES ('grep', 'linux', '1', 'grep [OPTION...] PATTERNS', 'print matching lines',
                     '/usr/share/man/man1/grep.1.gz', 'deadbeef', 1, datetime('now'))",
            [],
        )
        .unwrap();
        let command_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO flags (command_id, short, long, arg_type, arg_required, description,
                                 flag_group, source_line)
             VALUES (?1, '-r', '--recursive', NULL, 0, 'recurse', NULL, 168)",
            [command_id],
        )
        .unwrap();
        conn
    }

    #[test]
    fn detects_v1_before_migration_and_v2_after() {
        let conn = v1_db();
        assert_eq!(detect_version(&conn).unwrap(), 1);
        v1_to_v2(&conn).unwrap();
        assert_eq!(detect_version(&conn).unwrap(), 2);
    }

    #[test]
    fn migration_preserves_every_row_without_loss() {
        let conn = v1_db();
        v1_to_v2(&conn).unwrap();

        let commands: i64 = conn.query_row("SELECT count(*) FROM commands", [], |r| r.get(0)).unwrap();
        assert_eq!(commands, 1);
        let flags: i64 = conn.query_row("SELECT count(*) FROM flags", [], |r| r.get(0)).unwrap();
        assert_eq!(flags, 1);

        let (name, section, target_id): (String, String, i64) = conn
            .query_row("SELECT name, section, target_id FROM commands", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(name, "grep");
        assert_eq!(section, "1");

        let (target_name, exec_kind): (String, String) = conn
            .query_row("SELECT name, exec_kind FROM targets WHERE id = ?1", [target_id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(target_name, "grep");
        assert_eq!(exec_kind, "file");

        let short: String = conn
            .query_row(
                "SELECT short FROM flags WHERE command_id = (SELECT id FROM commands)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(short, "-r");
    }

    #[test]
    fn migrated_flags_have_an_empty_excerpt_rather_than_missing_column() {
        let conn = v1_db();
        v1_to_v2(&conn).unwrap();
        let excerpt: String = conn.query_row("SELECT excerpt FROM flags LIMIT 1", [], |r| r.get(0)).unwrap();
        assert_eq!(excerpt, "");
    }
}
