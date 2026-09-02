# Changelog

All notable changes to ShellIQ are documented here.

## 0.1.0-alpha.1 - 2026-09-01

First public alpha release.

- Build a local SQLite index from installed man pages without executing commands
  discovered during broad scans.
- Explain and validate command flags with citations from local documentation.
- Search flags by meaning and offer indexed Bash completion where the shell can
  safely provide a default completion policy.
- Provide opt-in Bash and Zsh widgets for editable suggestions and explanations.
- Include the optional `shelliq model setup|run|serve|doctor|remove` workflow.
  The standalone Rust CLI installs no AI dependencies unless setup is explicitly
  requested.
- Ship a documentation-conditioned CodeT5 ranker that passed its fresh
  command-disjoint and complete-abstention gates, with deterministic
  documentation fallback and Rust validation remaining authoritative.
- Ship checksum-protected Linux and macOS archives for x86-64 and ARM64.

Known alpha limitations:

- Option validity does not prove operands, pipelines, or command behavior match
  the user's intent.
- Native Windows is unsupported; Windows users should run ShellIQ in WSL2.
- The optional AI runtime supports x86-64/ARM64 Linux and Apple-silicon macOS;
  Intel macOS remains model-free.
- The model remains experimental. Its Python dependencies and weights are
  downloaded only after explicit setup.
