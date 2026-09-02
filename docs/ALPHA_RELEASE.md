# Optional model alpha

The public alpha ships one standalone Rust binary. Its index, search, explain,
completion, and documentation-fallback behavior requires no Python or model.
Nothing AI-related is installed or started automatically.

## Supported model hosts

The locked CPU runtime supports:

- Linux x86-64 and ARM64, including WSL2
- Apple-silicon macOS

The Rust CLI also ships for Intel macOS, but the optional model runtime does not
because the locked PyTorch release has no matching Intel macOS wheel. Native
Windows is unsupported; use WSL2.

## First use

Check the machine without installing anything:

```console
shelliq model doctor
```

If uv is absent, ShellIQ prints its official installation command and URL. Once
uv is available:

```console
shelliq model setup --scan
shelliq model run --json 'Show information about all CPUs'
```

Setup extracts and verifies the embedded recipe index and ranker head, asks uv
to acquire Python and install the locked CPU environment, downloads the exact
CodeT5 revision, and optionally builds the ordinary local man-page index. The
first setup downloads about 500 MB and needs no root access or Rust toolchain.

For repeated requests and the shell widgets:

```console
shelliq model serve
```

The adapter listens on numeric loopback only. ShellIQ treats its response as
untrusted: Rust parses, renders, reparses, checks local command and flag facts,
and applies deterministic policy before returning an editable command. Nothing
is executed automatically.

## Storage and removal

Default locations are:

- Runtime assets: `~/.local/share/shelliq/model/`
- Isolated environment, downloads, and extracted bootstrap: `~/.cache/shelliq/`
- Local command index: `~/.local/share/shelliq/index.sqlite`

`XDG_DATA_HOME`, `XDG_CACHE_HOME`, `SHELLIQ_MODEL_HOME`, and `SHELLIQ_INDEX` are
honored. Diagnose the resolved locations with `shelliq model doctor`.

Remove the optional model without touching the Rust CLI or command index:

```console
shelliq model remove
```

Automatic removal refuses a custom `SHELLIQ_MODEL_HOME`; unset it and remove the
custom directory deliberately if that override was used.

## Evidence and limitations

The documentation-conditioned v2 ranker passed its preregistered fresh-command
gate: 104/128 sufficient unseen commands compiled ready, all 128/128
insufficient-documentation pools abstained, and warm CPU p95 was 215 ms. See
[`documentation-cross-encoder-v2-results.md`](../training/experiments/documentation-cross-encoder-v2-results.md).

This remains an alpha. A locally valid command can still use the wrong operands,
combine incompatible stages, or fail to accomplish the request. Suggestions
remain editable and require review and an explicit Enter.

The older 0.5B GGUF and generative CodeT5 experiments are rejected development
baselines and are not shipped as the runtime. Their frozen results remain in
`training/experiments/` for auditability.
