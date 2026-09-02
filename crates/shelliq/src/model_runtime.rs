//! Extract and launch the optional pinned Python documentation ranker.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

const RUNTIME_VERSION: &str = "documentation-cross-encoder-v2";
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
        bail!("usage: shelliq model setup|serve|run [OPTIONS]");
    }
    let runtime = install_runtime()?;
    let executable = std::env::current_exe().context("locating current shelliq executable")?;
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
        .env("SHELLIQ_MODEL_SHELLIQ", executable)
        .env_remove("VIRTUAL_ENV")
        .status()
        .context("starting optional model runtime; install `uv` and try again")?;
    if !status.success() {
        bail!("shelliq model runtime exited with {status}");
    }
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
    Ok(cache_home()?.join("shelliq/model-venv"))
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

#[cfg(test)]
mod tests {
    use super::{FILES, RUNTIME_VERSION};

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
}
