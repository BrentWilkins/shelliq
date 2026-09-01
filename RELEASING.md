# Releasing ShellIQ

Releases are built by cargo-dist and GitHub Actions. A version tag creates a
prerelease when its semantic version contains a prerelease suffix.

## Prepare

1. Update the workspace version and internal path-dependency versions with
   Cargo, then refresh `Cargo.lock`.
2. Update `CHANGELOG.md` and the versioned installer URL in `README.md`.
3. Regenerate `.github/workflows/release.yml` with the cargo-dist version named
   in `dist-workspace.toml`:

   ```console
   dist generate
   dist plan
   ```

4. Run the same gates as CI:

   ```console
   cargo fmt --all --check
   cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
   cargo test --locked --workspace --all-features
   cargo package --locked --workspace
   bash shell/tests/run.bash
   shellcheck -x shell/shelliq.bash shell/tests/run.bash shell/tests/suggest-widget.bash
   uv run --frozen ruff check .
   uv run --frozen ruff format --check .
   ```

5. Build a host archive and inspect its contents before committing:

   ```console
   dist build --target x86_64-unknown-linux-gnu
   tar -tf target/distrib/shelliq-x86_64-unknown-linux-gnu.tar.xz
   ```

The archive must contain the binary, both licenses, the README, changelog, and
the Bash and Zsh integrations. The plan must contain only the `shelliq` app and
the four supported target triples.

## Publish

After the release commit passes CI, create and push its annotated version tag:

```console
git tag -a v0.1.0-alpha.1 -m 'shelliq 0.1.0-alpha.1'
git push origin v0.1.0-alpha.1
```

The release workflow builds native x86_64 and ARM64 archives on Linux and macOS,
generates per-artifact and unified SHA-256 checksums, creates the shell
installer, and publishes a GitHub prerelease. Download one archive after the
workflow completes, verify its checksum, and smoke-test `shelliq --version` and
`shelliq index build` on representative Linux and macOS hosts.
