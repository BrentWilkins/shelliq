# Changelog

All notable changes to ShellIQ are documented here.

## 0.1.0-alpha.1 - 2026-09-01

First public alpha release.

- Build a local SQLite index from installed man pages without executing
  commands discovered during broad scans.
- Explain and validate command flags with citations to local documentation.
- Search flags by meaning and offer indexed Bash completion where the shell can
  safely provide a default completion policy.
- Provide opt-in Bash and Zsh widgets for editable suggestions and explanations.
- Provide an experimental, feature-gated model suggestion path plus a
  deterministic documentation fallback; the default build remains model-free.
- Ship checksum-protected Linux and macOS archives for x86_64 and ARM64.

Known alpha limitations:

- Option validity does not prove that operands, pipelines, or command behavior
  match the user's intent.
- Native Windows is unsupported; Windows users should run ShellIQ in WSL2.
- The custom-model path remains experimental and is not included in prebuilt
  artifacts.
