//! Quote-aware scanning of a command line, and the constructs it refuses.
//!
//! P0 used `shlex::split` and turned its `None` into an empty vector, so an unbalanced
//! quote produced zero segments, zero findings, and exit 0 — fail-open on exactly the input
//! that deserves suspicion. `shlex` is also only a word splitter: it has no notion of
//! redirection, command substitution, or subshells, so `grep -r foo>out` came back as the
//! single operand `foo>out`, and `a|b` as one segment rather than two.
//!
//! This scanner is not a shell grammar either, and does not pretend to be one — adopting a
//! real parser is a dependency decision recorded in `PLAN.md`. What it does is *recognise
//! the constructs it cannot analyse and name them*, so the checker abstains instead of
//! guessing. Naming matters: "cannot parse that" is not an answer, but "this line contains
//! a redirection, which the checker does not analyse" tells the reader what to remove.

use std::fmt;

/// Shell syntax the checker does not analyse.
///
/// Every variant is a refusal, but they differ in blast radius, and the scanner rather than
/// the enum decides which applies. The structural ones make the whole line unreadable.
/// [`Unsupported::Expansion`] does not — the word is still a word, its run-time value is
/// simply unknown — so it costs only that word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// A quote that is never closed.
    UnbalancedQuote,
    /// A backslash at the very end of the input, escaping nothing.
    DanglingEscape,
    /// `>`, `<`, `>>`, `2>`, `&>`, and heredocs.
    Redirection,
    /// `$(...)` or a backtick pair. Arithmetic `$(( ))` is reported the same way: telling
    /// it apart from `$( ( ) )` needs a real parser, and both are refused anyway.
    CommandSubstitution,
    /// `<(...)` or `>(...)`.
    ProcessSubstitution,
    /// A parenthesised subshell.
    Subshell,
    /// `{ ...; }` as a grouping construct, as opposed to `{a,b}` brace expansion.
    BraceGroup,
    /// `${` with no closing brace.
    UnterminatedExpansion,
    /// `$VAR`, `${VAR}`, `$1`, `$'...'`. Not a syntax error — the value is simply not known
    /// until the line runs, and it may expand to flags, so it cannot be waved through as an
    /// operand.
    Expansion,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Unsupported::UnbalancedQuote => "an unbalanced quote",
            Unsupported::DanglingEscape => "a trailing backslash",
            Unsupported::Redirection => "a redirection",
            Unsupported::CommandSubstitution => "a command substitution",
            Unsupported::ProcessSubstitution => "a process substitution",
            Unsupported::Subshell => "a subshell",
            Unsupported::BraceGroup => "a brace group",
            Unsupported::UnterminatedExpansion => "an unterminated expansion",
            Unsupported::Expansion => "an expansion, resolved only when the line runs",
        };
        f.write_str(text)
    }
}

/// One word of a command line, after quote removal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// The word with quoting removed. Expansions are kept in their source form, so
    /// `--out=$dest` stays readable rather than collapsing to `--out=`.
    pub text: String,
    /// Set when the word contains an expansion, so its value is unknown here.
    pub opaque: bool,
}

impl Word {
    /// Convenience for tests and callers that only care about the literal text.
    pub fn literal(text: &str) -> Self {
        Word {
            text: text.to_string(),
            opaque: false,
        }
    }
}

/// One `cmd ...` stage of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub command: Word,
    pub args: Vec<Word>,
}

/// Split a command line into pipeline segments.
///
/// Returns `Err` when the line contains syntax the checker cannot analyse. Callers must
/// surface that as an abstention; treating it as an empty line is the bug this replaced.
pub fn segments(line: &str) -> Result<Vec<Segment>, Unsupported> {
    let chars: Vec<char> = line.chars().collect();
    let mut b = Builder::default();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '\\' => {
                let Some(&next) = chars.get(i + 1) else {
                    return Err(Unsupported::DanglingEscape);
                };
                // A backslash-newline is a line continuation and contributes nothing.
                if next != '\n' {
                    b.push_char(next);
                }
                i += 2;
            }
            '\'' => {
                b.begin();
                i += 1;
                loop {
                    let Some(&c) = chars.get(i) else {
                        return Err(Unsupported::UnbalancedQuote);
                    };
                    i += 1;
                    if c == '\'' {
                        break;
                    }
                    b.push_char(c);
                }
            }
            '"' => {
                b.begin();
                i += 1;
                loop {
                    let Some(&c) = chars.get(i) else {
                        return Err(Unsupported::UnbalancedQuote);
                    };
                    match c {
                        '"' => {
                            i += 1;
                            break;
                        }
                        // Substitution and expansion stay live inside double quotes.
                        '`' => return Err(Unsupported::CommandSubstitution),
                        '$' => i = expansion(&chars, i, &mut b)?,
                        '\\' => {
                            let Some(&next) = chars.get(i + 1) else {
                                return Err(Unsupported::UnbalancedQuote);
                            };
                            // Only four characters stay escapable inside double quotes;
                            // before anything else the backslash is itself literal.
                            if matches!(next, '$' | '`' | '"' | '\\') {
                                b.push_char(next);
                                i += 2;
                            } else if next == '\n' {
                                i += 2;
                            } else {
                                b.push_char('\\');
                                i += 1;
                            }
                        }
                        _ => {
                            b.push_char(c);
                            i += 1;
                        }
                    }
                }
            }
            '`' => return Err(Unsupported::CommandSubstitution),
            '$' => i = expansion(&chars, i, &mut b)?,
            '<' | '>' => {
                return Err(if chars.get(i + 1) == Some(&'(') {
                    Unsupported::ProcessSubstitution
                } else {
                    Unsupported::Redirection
                });
            }
            '(' | ')' => return Err(Unsupported::Subshell),
            // A brace only groups commands when it stands as a word of its own; anywhere
            // else it is brace expansion, which is an operand concern and harmless here.
            '{' if !b.in_word() && chars.get(i + 1).is_none_or(|c| c.is_whitespace()) => {
                return Err(Unsupported::BraceGroup);
            }
            '}' if !b.in_word() && chars.get(i + 1).is_none_or(|c| c.is_whitespace() || *c == ';') => {
                return Err(Unsupported::BraceGroup);
            }
            '|' => {
                b.end_segment();
                i += if chars.get(i + 1) == Some(&'|') { 2 } else { 1 };
            }
            '&' => {
                if chars.get(i + 1) == Some(&'>') {
                    return Err(Unsupported::Redirection);
                }
                b.end_segment();
                i += if chars.get(i + 1) == Some(&'&') { 2 } else { 1 };
            }
            ';' | '\n' => {
                b.end_segment();
                i += 1;
            }
            c if c.is_whitespace() => {
                b.end_word();
                i += 1;
            }
            c => {
                b.push_char(c);
                i += 1;
            }
        }
    }

    Ok(b.finish())
}

/// Consume the expansion beginning at the `$` in `chars[i]`, returning the index past it.
fn expansion(chars: &[char], i: usize, b: &mut Builder) -> Result<usize, Unsupported> {
    match chars.get(i + 1) {
        Some('(') => Err(Unsupported::CommandSubstitution),
        Some('{') => {
            let mut depth = 0usize;
            for (offset, c) in chars[i + 1..].iter().enumerate() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            let end = i + 2 + offset;
                            b.push_expansion(&chars[i..end]);
                            return Ok(end);
                        }
                    }
                    _ => {}
                }
            }
            Err(Unsupported::UnterminatedExpansion)
        }
        // `$'...'` and `$"..."`: the quote itself is left for the main loop to handle, but
        // the word is opaque because the escapes inside are not the ordinary ones.
        Some('\'' | '"') => {
            b.push_expansion(&chars[i..i + 1]);
            Ok(i + 1)
        }
        Some(&c) if c.is_alphabetic() || c == '_' => {
            let mut end = i + 2;
            while chars.get(end).is_some_and(|c| c.is_alphanumeric() || *c == '_') {
                end += 1;
            }
            b.push_expansion(&chars[i..end]);
            Ok(end)
        }
        // Single-character special parameters: `$1`, `$?`, `$@`, `$$`, and friends.
        Some(&c) if c.is_ascii_digit() || matches!(c, '?' | '!' | '@' | '*' | '#' | '-' | '$') => {
            b.push_expansion(&chars[i..i + 2]);
            Ok(i + 2)
        }
        // A `$` before anything else is an ordinary character, as in `awk '{print $}'`.
        _ => {
            b.push_char('$');
            Ok(i + 1)
        }
    }
}

/// Accumulates words into segments.
///
/// `started` is tracked separately from `text` being non-empty so that `grep ""` yields an
/// empty word rather than no word at all.
#[derive(Default)]
struct Builder {
    segments: Vec<Segment>,
    words: Vec<Word>,
    text: String,
    started: bool,
    opaque: bool,
}

impl Builder {
    fn begin(&mut self) {
        self.started = true;
    }

    fn in_word(&self) -> bool {
        self.started
    }

    fn push_char(&mut self, c: char) {
        self.text.push(c);
        self.started = true;
    }

    fn push_expansion(&mut self, raw: &[char]) {
        self.text.extend(raw);
        self.started = true;
        self.opaque = true;
    }

    fn end_word(&mut self) {
        if !self.started {
            return;
        }
        self.words.push(Word {
            text: std::mem::take(&mut self.text),
            opaque: self.opaque,
        });
        self.started = false;
        self.opaque = false;
    }

    fn end_segment(&mut self) {
        self.end_word();
        if self.words.is_empty() {
            return;
        }
        let mut words = std::mem::take(&mut self.words);
        let command = words.remove(0);
        self.segments.push(Segment { command, args: words });
    }

    fn finish(mut self) -> Vec<Segment> {
        self.end_segment();
        self.segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<String> {
        segments(line)
            .unwrap()
            .into_iter()
            .flat_map(|s| std::iter::once(s.command).chain(s.args).map(|w| w.text).collect::<Vec<_>>())
            .collect()
    }

    #[test]
    fn quoting_is_removed_and_respected() {
        assert_eq!(words(r#"grep -r "a b" 'c d'"#), ["grep", "-r", "a b", "c d"]);
    }

    #[test]
    fn an_empty_quoted_word_is_still_a_word() {
        assert_eq!(words(r#"grep -r "" ."#), ["grep", "-r", "", "."]);
    }

    #[test]
    fn backslash_escapes_the_next_character() {
        assert_eq!(words(r"grep -r a\ b"), ["grep", "-r", "a b"]);
    }

    #[test]
    fn inside_double_quotes_a_backslash_is_mostly_literal() {
        assert_eq!(words(r#"grep "a\nb" "c\"d""#), ["grep", r"a\nb", r#"c"d"#]);
    }

    /// P0 only saw separators as isolated tokens, so `a|b` was one segment.
    #[test]
    fn separators_are_recognised_when_attached() {
        let segs = segments("grep -r x|head -5").unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].command.text, "grep");
        assert_eq!(segs[1].command.text, "head");
    }

    #[test]
    fn every_separator_ends_a_segment() {
        for line in ["a | b", "a || b", "a && b", "a; b", "a & b", "a\nb"] {
            assert_eq!(segments(line).unwrap().len(), 2, "line {line:?}");
        }
    }

    #[test]
    fn a_quoted_pipe_is_not_a_separator() {
        assert_eq!(segments(r#"grep -r "a | b" ."#).unwrap().len(), 1);
    }

    #[test]
    fn unbalanced_quotes_are_refused_rather_than_swallowed() {
        assert_eq!(segments(r#"grep -r "unclosed"#), Err(Unsupported::UnbalancedQuote));
        assert_eq!(segments("grep -r 'unclosed"), Err(Unsupported::UnbalancedQuote));
    }

    #[test]
    fn a_dangling_backslash_is_refused() {
        assert_eq!(segments(r"grep -r \"), Err(Unsupported::DanglingEscape));
    }

    #[test]
    fn redirections_are_refused_however_they_are_spelled() {
        for line in [
            "grep -r foo > out",
            "grep -r foo>out",
            "grep -r foo >> out",
            "grep -r foo 2>/dev/null",
            "grep -r foo &> out",
            "cat < in",
            "cat <<EOF",
        ] {
            assert_eq!(segments(line), Err(Unsupported::Redirection), "line {line:?}");
        }
    }

    #[test]
    fn substitutions_and_subshells_are_refused() {
        assert_eq!(segments("grep -r $(cat f)"), Err(Unsupported::CommandSubstitution));
        assert_eq!(segments("grep -r `cat f`"), Err(Unsupported::CommandSubstitution));
        assert_eq!(segments(r#"grep -r "$(cat f)""#), Err(Unsupported::CommandSubstitution));
        assert_eq!(segments("echo $((1 + 1))"), Err(Unsupported::CommandSubstitution));
        assert_eq!(segments("diff <(a) <(b)"), Err(Unsupported::ProcessSubstitution));
        assert_eq!(segments("(cd /tmp && ls)"), Err(Unsupported::Subshell));
    }

    #[test]
    fn a_brace_group_is_refused_but_brace_expansion_is_not() {
        assert_eq!(segments("{ ls; }"), Err(Unsupported::BraceGroup));
        assert_eq!(words("cp a.{c,h} dir"), ["cp", "a.{c,h}", "dir"]);
    }

    /// Single quotes neutralise everything, and the scanner must agree.
    #[test]
    fn single_quoted_metacharacters_are_ordinary_text() {
        assert_eq!(
            words(r#"grep -r 'a > b (c) `d` $(e) | f' ."#),
            ["grep", "-r", "a > b (c) `d` $(e) | f", "."]
        );
        assert_eq!(words(r"grep -r a\>b"), ["grep", "-r", "a>b"]);
    }

    /// Double quotes neutralise redirection and grouping but *not* substitution, which is
    /// why `"$(cat f)"` is refused above while `"a > b"` is not.
    #[test]
    fn double_quotes_neutralise_everything_but_substitution() {
        assert_eq!(
            words(r#"grep -r "a > b (c) | d; e" ."#),
            ["grep", "-r", "a > b (c) | d; e", "."]
        );
    }

    #[test]
    fn expansions_mark_their_word_opaque_without_failing_the_line() {
        let segs = segments("grep $OPTS ${pattern} $1 .").unwrap();
        let args = &segs[0].args;
        assert_eq!(
            args.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["$OPTS", "${pattern}", "$1", "."]
        );
        assert_eq!(args.iter().map(|w| w.opaque).collect::<Vec<_>>(), [true, true, true, false]);
    }

    #[test]
    fn an_expansion_inside_a_word_makes_the_whole_word_opaque() {
        let segs = segments("curl --output=$dest url").unwrap();
        assert_eq!(segs[0].args[0].text, "--output=$dest");
        assert!(segs[0].args[0].opaque);
        assert!(!segs[0].args[1].opaque);
    }

    #[test]
    fn an_unterminated_expansion_is_refused() {
        assert_eq!(segments("grep ${pattern"), Err(Unsupported::UnterminatedExpansion));
    }

    #[test]
    fn a_lone_dollar_is_an_ordinary_character() {
        let segs = segments("grep -r 'x' $").unwrap();
        assert_eq!(segs[0].args[2], Word::literal("$"));
    }
}
