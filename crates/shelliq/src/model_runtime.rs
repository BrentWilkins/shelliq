//! Extract and launch the optional pinned Python documentation ranker.

use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

const RUNTIME_VERSION: &str = "documentation-cross-encoder-v2";
const MODEL_REVISION: &str = "b1ee9570c289f21b5922b9c768a1ce12957bf968";
const UV_INSTALL_URL: &str = "https://docs.astral.sh/uv/getting-started/installation/";
const UV_INSTALL_COMMAND: &str = "curl -LsSf https://astral.sh/uv/install.sh | sh";
const FILES: &[(&str, &[u8])] = &[
    ("pyproject.toml", include_bytes!("../../../training/pyproject.toml")),
    ("uv.lock", include_bytes!("../../../training/uv.lock")),
    (
        "shelliq_training/__init__.py",
        include_bytes!("../../../training/shelliq_training/__init__.py"),
    ),
    (
        "shelliq_training/data.py",
        include_bytes!("../../../training/shelliq_training/data.py"),
    ),
    (
        "shelliq_training/prompt.py",
        include_bytes!("../../../training/shelliq_training/prompt.py"),
    ),
    (
        "shelliq_training/documentation_templates.py",
        include_bytes!("../../../training/shelliq_training/documentation_templates.py"),
    ),
    (
        "shelliq_training/semantic_actions.py",
        include_bytes!("../../../training/shelliq_training/semantic_actions.py"),
    ),
    (
        "shelliq_training/semantic_equivalence.py",
        include_bytes!("../../../training/shelliq_training/semantic_equivalence.py"),
    ),
    (
        "shelliq_training/documentation_cross_encoder.py",
        include_bytes!("../../../training/shelliq_training/documentation_cross_encoder.py"),
    ),
    (
        "shelliq_training/model_runtime.py",
        include_bytes!("../../../training/shelliq_training/model_runtime.py"),
    ),
    (
        "shelliq_training/assets/documentation-index-v2.jsonl.xz",
        include_bytes!("../../../training/shelliq_training/assets/documentation-index-v2.jsonl.xz"),
    ),
    (
        "shelliq_training/assets/documentation-cross-encoder-v2.pt",
        include_bytes!("../../../training/shelliq_training/assets/documentation-cross-encoder-v2.pt"),
    ),
    (
        "shelliq_training/assets/TLDR-NOTICE.md",
        include_bytes!("../../../training/shelliq_training/assets/TLDR-NOTICE.md"),
    ),
];

pub fn run(arguments: &[OsString]) -> Result<()> {
    if arguments.is_empty() {
        bail!("usage: shelliq model setup|serve|run|doctor|remove [OPTIONS]");
    }
    let command = arguments[0].to_string_lossy();
    if arguments.len() == 2 && (arguments[1] == "--help" || arguments[1] == "-h") {
        return print_help(&command);
    }
    match command.as_ref() {
        "doctor" if arguments.len() == 1 => return doctor(),
        "remove" if arguments.len() == 1 => return remove(),
        "doctor" | "remove" => bail!("`shelliq model {command}` does not accept additional arguments"),
        "setup" | "serve" | "run" => {}
        _ => bail!("unknown model command `{command}`; expected setup, serve, run, doctor, or remove"),
    }
    ensure_supported_platform()?;
    ensure_uv()?;
    let runtime = install_runtime()?;
    let executable = std::env::current_exe().context("locating current shelliq executable")?;
    let cache = shelliq_cache_home()?;
    if command == "setup" {
        eprintln!("Preparing ShellIQ's isolated model runtime; the first setup downloads about 500 MB.");
    }
    let status = Command::new("uv")
        .arg("run")
        .arg("--frozen")
        .arg("--project")
        .arg(&runtime)
        .arg("--no-dev")
        .arg("--no-group")
        .arg("training")
        .arg("python")
        .arg("-m")
        .arg("shelliq_training.model_runtime")
        .args(arguments)
        .env("PYTHONPATH", &runtime)
        .env("UV_PROJECT_ENVIRONMENT", runtime_environment()?)
        .env("UV_CACHE_DIR", cache.join("uv-cache"))
        .env("HF_HOME", cache.join("huggingface"))
        .env("SHELLIQ_MODEL_SHELLIQ", executable)
        .env_remove("VIRTUAL_ENV")
        .status()
        .context("starting optional model runtime; install `uv` and try again")?;
    if !status.success() {
        bail!("shelliq model runtime exited with {status}");
    }
    Ok(())
}

fn ensure_uv() -> Result<()> {
    match Command::new("uv").arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => bail!(
            "`uv --version` failed with {}. Reinstall uv from {UV_INSTALL_URL}",
            output.status
        ),
        Err(error) if error.kind() == ErrorKind::NotFound => bail!(uv_missing_message()),
        Err(error) => Err(error).context("checking the uv installation"),
    }
}

fn uv_missing_message() -> String {
    format!(
        "uv is required only for optional AI setup and was not found.\n\
         Install it on macOS, Linux, or WSL2 with:\n\n    {UV_INSTALL_COMMAND}\n\n\
         Then open a new shell and rerun:\n\n    shelliq model setup --scan\n\n\
         Official instructions: {UV_INSTALL_URL}\n\
         Normal model-free ShellIQ commands work without uv."
    )
}

fn ensure_supported_platform() -> Result<()> {
    let supported = matches!(
        (std::env::consts::OS, std::env::consts::ARCH),
        ("linux", "x86_64" | "aarch64") | ("macos", "aarch64")
    );
    if !supported {
        bail!(
            "the optional model runtime does not currently support {}/{}; model-free ShellIQ remains available",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    Ok(())
}

fn doctor() -> Result<()> {
    let cache = shelliq_cache_home()?;
    let model = model_home()?;
    let command_index = command_index_path()?;
    let runtime = cache.join("runtime").join(RUNTIME_VERSION);
    let environment = cache.join("model-venv");
    let weights = cache
        .join("huggingface/hub/models--Salesforce--codet5-small/snapshots")
        .join(MODEL_REVISION)
        .join("pytorch_model.bin");
    let uv = match Command::new("uv").arg("--version").output() {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        _ => "not found".to_owned(),
    };
    let supported = ensure_supported_platform().is_ok();
    println!("ShellIQ optional model diagnostics");
    println!(
        "platform: {}/{} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if supported { "supported" } else { "unsupported" }
    );
    println!("uv: {uv}");
    report_path("embedded runtime extraction", &runtime);
    report_path("isolated Python environment", &environment);
    report_path("ranker assets", &model.join("documentation-cross-encoder-v2.pt"));
    report_path("pinned CodeT5 weights", &weights);
    report_path("local command index", &command_index);
    if !supported {
        println!("next: continue with model-free ShellIQ on this platform");
    } else if uv == "not found" {
        println!("next: install uv with `{UV_INSTALL_COMMAND}`, then run `shelliq model setup --scan`");
        println!("docs: {UV_INSTALL_URL}");
    } else if !weights.is_file() || !model.join("documentation-cross-encoder-v2.pt").is_file() {
        println!("next: shelliq model setup --scan");
    } else if !command_index.is_file() {
        println!("next: shelliq index scan");
    } else {
        println!("next: shelliq model run --json 'Show information about all CPUs'");
    }
    Ok(())
}

fn remove() -> Result<()> {
    if std::env::var_os("SHELLIQ_MODEL_HOME").is_some() {
        bail!("SHELLIQ_MODEL_HOME is set; refusing automatic removal of a custom path. Unset it to remove the default runtime.");
    }
    let cache = shelliq_cache_home()?;
    let model = model_home()?;
    for path in [
        cache.join("runtime").join(RUNTIME_VERSION),
        cache.join("model-venv"),
        cache.join("huggingface"),
        cache.join("uv-cache"),
        model,
    ] {
        remove_owned_directory(&path)?;
    }
    println!("Removed the optional ShellIQ model runtime and downloads.");
    println!("The Rust CLI and local command index were preserved.");
    println!("Run `shelliq model setup --scan` to install the model again.");
    Ok(())
}

fn remove_owned_directory(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    fs::remove_dir_all(path).with_context(|| format!("removing {}", path.display()))
}

fn report_path(label: &str, path: &Path) {
    println!("{label}: {} ({})", path.display(), state(path.exists()));
}

fn state(value: bool) -> &'static str {
    if value { "ready" } else { "not installed" }
}

fn print_help(command: &str) -> Result<()> {
    let help = match command {
        "setup" => {
            "Usage: shelliq model setup [--offline] [--scan] [--home PATH] [--index PATH] [--shelliq PATH]\n\
             Installs the isolated runtime and pinned model; --scan also builds the local command index."
        }
        "serve" => {
            "Usage: shelliq model serve [--home PATH] [--host 127.0.0.1|::1] [--port 8080] [--device auto|cpu|cuda]\n\
             Runs the warmed loopback adapter for repeated requests and shell widgets."
        }
        "run" => {
            "Usage: shelliq model run [--json] [--home PATH] [--index PATH] [--shelliq PATH] [--device auto|cpu|cuda] [--timeout-ms N] INSTRUCTION...\n\
             Starts a temporary adapter, requests one suggestion, and shuts it down."
        }
        "doctor" => "Usage: shelliq model doctor\nReports dependencies, platform support, installed assets, and the next step.",
        "remove" => {
            "Usage: shelliq model remove\nRemoves only the optional runtime and downloads; preserves the Rust CLI and command index."
        }
        _ => bail!("unknown model command `{command}`"),
    };
    println!("{help}");
    Ok(())
}

fn install_runtime() -> Result<PathBuf> {
    let root = cache_home()?.join("shelliq/runtime").join(RUNTIME_VERSION);
    for (relative, contents) in FILES {
        let destination = root.join(relative);
        if fs::read(&destination).ok().as_deref() == Some(*contents) {
            continue;
        }
        let parent = destination.parent().context("runtime bundle path has no parent")?;
        fs::create_dir_all(parent).with_context(|| format!("creating model runtime directory {}", parent.display()))?;
        let temporary = destination.with_extension(format!("shelliq-new-{}", std::process::id()));
        fs::write(&temporary, contents).with_context(|| format!("writing model runtime file {}", temporary.display()))?;
        fs::rename(&temporary, &destination)
            .with_context(|| format!("installing model runtime file {}", destination.display()))?;
    }
    Ok(root)
}

fn runtime_environment() -> Result<PathBuf> {
    Ok(shelliq_cache_home()?.join("model-venv"))
}

fn cache_home() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .context("HOME is unset; set XDG_CACHE_HOME so ShellIQ can install the optional model runtime")?;
    Ok(Path::new(&home).join(".cache"))
}

fn shelliq_cache_home() -> Result<PathBuf> {
    Ok(cache_home()?.join("shelliq"))
}

fn data_home() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .context("HOME is unset; set XDG_DATA_HOME for ShellIQ data")?;
    Ok(Path::new(&home).join(".local/share"))
}

fn model_home() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("SHELLIQ_MODEL_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    Ok(data_home()?.join("shelliq/model"))
}

fn command_index_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("SHELLIQ_INDEX").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    Ok(data_home()?.join("shelliq/index.sqlite"))
}

#[cfg(test)]
mod tests {
    use super::{
        FILES, RUNTIME_VERSION, UV_INSTALL_COMMAND, UV_INSTALL_URL, print_help, remove_owned_directory, state, uv_missing_message,
    };

    #[test]
    fn embedded_runtime_has_every_required_file() {
        assert_eq!(RUNTIME_VERSION, "documentation-cross-encoder-v2");
        assert!(FILES.iter().any(|(path, _)| *path == "uv.lock"));
        assert!(
            FILES.iter().any(
                |(path, contents)| *path == "shelliq_training/assets/documentation-index-v2.jsonl.xz" && !contents.is_empty()
            )
        );
    }

    #[test]
    fn model_help_is_available_without_starting_python() {
        print_help("setup").unwrap();
        print_help("doctor").unwrap();
        assert!(print_help("unknown").is_err());
        assert_eq!(state(true), "ready");
        assert_eq!(state(false), "not installed");
    }

    #[test]
    fn missing_uv_message_is_actionable_and_keeps_core_available() {
        let message = uv_missing_message();
        assert!(message.contains(UV_INSTALL_COMMAND));
        assert!(message.contains(UV_INSTALL_URL));
        assert!(message.contains("shelliq model setup --scan"));
        assert!(message.contains("model-free ShellIQ commands work without uv"));
    }

    #[test]
    fn owned_runtime_directory_removal_is_scoped() {
        let parent = std::env::temp_dir().join(format!("shelliq-model-remove-test-{}", std::process::id()));
        let runtime = parent.join("model-venv");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(parent.join("preserved"), b"keep").unwrap();
        remove_owned_directory(&runtime).unwrap();
        assert!(!runtime.exists());
        assert!(parent.join("preserved").is_file());
        std::fs::remove_dir_all(parent).unwrap();
    }
}
