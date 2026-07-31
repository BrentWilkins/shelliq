-- shelliq index schema.
--
-- Text comparison is SQLite's default BINARY collation throughout. This is load-bearing,
-- not incidental: `flags.short = '-r'` must not match '-R'. No column in this file may be
-- given COLLATE NOCASE, and `flags_case_is_binary` in lib.rs guards that.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One resolved executable identity: the file (or builtin) a shell would actually run for
-- `name`, right now. Two installs of the same command name never share a row, because they
-- are not the same fact source even though a bare command lookup would use the same name.
CREATE TABLE IF NOT EXISTS targets (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    platform     TEXT NOT NULL,
    -- 'file' | 'builtin' | 'absent'.
    exec_kind    TEXT NOT NULL,
    -- Absolute path, for exec_kind = 'file'; NULL otherwise.
    exec_path    TEXT,
    -- sha256 of the file's contents as of the last harvest or refresh. NULL until computed.
    exec_hash    TEXT,
    -- Cheap identity signals, checked at lookup time without re-reading the file.
    exec_size    INTEGER,
    exec_mtime   INTEGER,
    first_seen   TEXT NOT NULL,
    last_checked TEXT NOT NULL
);

-- Not a UNIQUE constraint: SQLite treats every NULL exec_path as distinct, which would let
-- duplicate builtin/absent rows slip past it silently. Index::insert_command enforces
-- identity itself with an explicit SELECT before insert or update.
CREATE INDEX IF NOT EXISTS targets_identity ON targets (name, exec_kind, exec_path);

CREATE TABLE IF NOT EXISTS commands (
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
    -- Parser version is stored per row so a parser fix invalidates previously harvested
    -- rows even when the source file is byte-identical.
    parser_version INTEGER NOT NULL DEFAULT 0,
    harvested_at   TEXT NOT NULL,
    UNIQUE (target_id, section)
);

CREATE TABLE IF NOT EXISTS subcommands (
    id         INTEGER PRIMARY KEY,
    command_id INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
    -- Full path as typed: 'ollama list', 'git remote add'.
    path       TEXT NOT NULL,
    summary    TEXT NOT NULL DEFAULT '',
    UNIQUE (command_id, path)
);

CREATE TABLE IF NOT EXISTS flags (
    id            INTEGER PRIMARY KEY,
    command_id    INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
    subcommand_id INTEGER REFERENCES subcommands (id) ON DELETE CASCADE,
    -- Exact spelling with the leading dash, case preserved.
    short         TEXT,
    long          TEXT,
    arg_type      TEXT,
    arg_required  INTEGER NOT NULL DEFAULT 1,
    description   TEXT NOT NULL DEFAULT '',
    flag_group    TEXT,
    source_line   INTEGER NOT NULL DEFAULT 0,
    -- The tag and body lines exactly as rendered, quoted verbatim by `shelliq source`.
    excerpt       TEXT NOT NULL DEFAULT '',
    -- Populated from local shell history frequency; drives completion ordering only.
    rank_personal INTEGER NOT NULL DEFAULT 0,
    -- Set when the flag appears in a tldr example.
    rank_tldr     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS flags_by_short ON flags (command_id, short);
CREATE INDEX IF NOT EXISTS flags_by_long  ON flags (command_id, long);
CREATE INDEX IF NOT EXISTS commands_by_name ON commands (name, platform);
CREATE INDEX IF NOT EXISTS commands_by_target ON commands (target_id);

CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY,
    command_id  INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT ''
);

-- Which flags an example is actually about, not merely the whole example line. A tldr
-- example mentions several flags at once ("curl -L -D - url"), and prose matched via
-- examples_fts cannot say by itself which flag it boosts; this join is what makes that
-- explicit. Only populated for flags confirmed to exist on this machine's target — see
-- Index::insert_tldr_examples.
CREATE TABLE IF NOT EXISTS example_flags (
    example_id INTEGER NOT NULL REFERENCES examples (id) ON DELETE CASCADE,
    flag_id    INTEGER NOT NULL REFERENCES flags (id) ON DELETE CASCADE,
    PRIMARY KEY (example_id, flag_id)
);

CREATE INDEX IF NOT EXISTS example_flags_by_flag ON example_flags (flag_id);

-- Flag-to-flag references mined from descriptions: curl's --location-trusted reads "Like
-- -L, --location, but...". Extraction happens once at ingestion time (see
-- Index::insert_command / mentioned_flag_indices) rather than at query time, because prose
-- mentions a flag for many reasons besides being related to it and the edges deserve their
-- own precision measurement, not a live text scan. Expansion through this table is capped
-- at one hop — see Index::search_flags_via_edges.
CREATE TABLE IF NOT EXISTS flag_edges (
    from_flag_id INTEGER NOT NULL REFERENCES flags (id) ON DELETE CASCADE,
    to_flag_id   INTEGER NOT NULL REFERENCES flags (id) ON DELETE CASCADE,
    PRIMARY KEY (from_flag_id, to_flag_id)
);

CREATE INDEX IF NOT EXISTS flag_edges_by_from ON flag_edges (from_flag_id);

-- Description search is what turns "follow redirect" into `-L, --location`. Without it,
-- a 258-flag page is only navigable by someone who already knows the flag name.
CREATE VIRTUAL TABLE IF NOT EXISTS flags_fts USING fts5 (
    description,
    spelling,
    content = 'flags',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS flags_fts_insert AFTER INSERT ON flags BEGIN
    INSERT INTO flags_fts (rowid, description, spelling)
    VALUES (new.id, new.description, coalesce(new.short, '') || ' ' || coalesce(new.long, ''));
END;

CREATE TRIGGER IF NOT EXISTS flags_fts_delete AFTER DELETE ON flags BEGIN
    INSERT INTO flags_fts (flags_fts, rowid, description, spelling)
    VALUES ('delete', old.id, old.description, coalesce(old.short, '') || ' ' || coalesce(old.long, ''));
END;

CREATE VIRTUAL TABLE IF NOT EXISTS examples_fts USING fts5 (
    text,
    description,
    content = 'examples',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS examples_fts_insert AFTER INSERT ON examples BEGIN
    INSERT INTO examples_fts (rowid, text, description)
    VALUES (new.id, new.text, new.description);
END;

CREATE TRIGGER IF NOT EXISTS examples_fts_delete AFTER DELETE ON examples BEGIN
    INSERT INTO examples_fts (examples_fts, rowid, text, description)
    VALUES ('delete', old.id, old.text, old.description);
END;
