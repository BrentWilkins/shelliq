#!/usr/bin/env bash

set -euo pipefail

bash -n shell/shelliq.bash shell/tests/run.bash shell/tests/suggest-widget.bash
bash shell/tests/suggest-widget.bash
zsh -n shell/shelliq.zsh shell/tests/suggest-widget.zsh
zsh shell/tests/suggest-widget.zsh
