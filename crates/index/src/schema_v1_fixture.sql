-- shelliq index schema.
--
-- Text comparison is SQLite's default BINARY collation throughout. This is load-bearing,
-- not incidental: `flags.short = '-r'` must not match '-R'. No column in this file may be
-- given COLLATE NOCASE, and `flags_case_is_binary` in lib.rs guards that.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS commands (
    id             INTEGER PRIMARY KEY,
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
    UNIQUE (name, platform, section)
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
    -- Populated from local shell history frequency; drives completion ordering only.
    rank_personal INTEGER NOT NULL DEFAULT 0,
    -- Set when the flag appears in a tldr example.
    rank_tldr     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS flags_by_short ON flags (command_id, short);
CREATE INDEX IF NOT EXISTS flags_by_long  ON flags (command_id, long);
CREATE INDEX IF NOT EXISTS commands_by_name ON commands (name, platform);

CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY,
    command_id  INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT ''
);

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
