# shelliq.zsh — zsh integration. Suggestions are opt-in and never auto-executed.
#
# Two things, both designed not to fight oh-my-zsh, zsh-autosuggestions, or
# zsh-syntax-highlighting:
#
#   C-x C-h   Explain the current buffer: every flag checked against the local index,
#             printed below the prompt. Never touches $BUFFER, so there is nothing to undo.
#
#   Tab       A fallback completer for flags. Registered first in the `completer` list, but
#             it declines immediately for any command with a native completion function
#             (`_git`, `_docker`, ...), so it only ever fires for commands zsh would
#             otherwise fall back to plain filename completion for.
#
# Model suggestions are opt-in through the widget below.

_shelliq_explain_widget() {
    emulate -L zsh
    if [[ -z $BUFFER ]]; then
        zle -M 'shelliq: buffer is empty'
        return
    fi
    local out
    out=$(shelliq explain -- "$BUFFER" 2>&1)
    zle -I
    zle -R
    print
    print -r -- "$out"
    zle reset-prompt
}
zle -N _shelliq_explain_widget
bindkey '^X^H' _shelliq_explain_widget

# C-x C-g turns the current English buffer into an editable suggestion. It never
# accepts or executes the result; Enter remains an explicit user action.
_shelliq_suggest_widget() {
  emulate -L zsh
  if [[ -z $BUFFER ]]; then
    zle -M 'shelliq: type a request first'
    return
  fi

  local request=$BUFFER out command
  local -a lines
  out=$(shelliq suggest -- "$request" 2>&1)
  local status=$?
  lines=("${(@f)out}")
  if (( status != 0 || ${#lines} == 0 )); then
    zle -I
    print
    print -r -- "$out"
    zle reset-prompt
    return
  fi

  command=${lines[-1]}
  lines[-1]=()
  zle split-undo
  BUFFER=$command
  CURSOR=${#BUFFER}
  zle -I
  if (( ${#lines} )); then
    print
    print -r -- "${(F)lines}"
  fi
  zle reset-prompt
}
zle -N _shelliq_suggest_widget
bindkey '^X^G' _shelliq_suggest_widget

# Function form: `shelliq-suggest find large files`, then edit/run the preloaded line.
# `print -z` writes to zsh's input buffer and never executes it.
shelliq-suggest() {
  emulate -L zsh
  if (( $# == 0 )); then
    print -u2 -- 'usage: shelliq-suggest <natural-language request>'
    return 2
  fi
  local out command
  local -a lines
  out=$(shelliq suggest -- "$@" 2>&1) || {
    print -u2 -r -- "$out"
    return 1
  }
  lines=("${(@f)out}")
  command=${lines[-1]}
  lines[-1]=()
  (( ${#lines} )) && print -u2 -r -- "${(F)lines}"
  print -z -- "$command"
}

# A completion function, not a widget: registered onto the `completer` style's list rather
# than bound to Tab directly. `_complete` (the standard completer) can't be trusted to
# "produce nothing" for commands with no dedicated completion function — it falls back to
# plain filename completion itself, which counts as success and would starve `_shelliq` if
# `_shelliq` ran after it. So `_shelliq` runs first and does its own check instead: it
# declines immediately if a native completion function is already registered for the
# command (`${(k)_comps[$cmd]}`), leaving `git`, `docker`, and friends to `_complete` as
# normal, and only offers anything for commands with no completion of their own.
_shelliq() {
    emulate -L zsh
    local -a words
    words=(${(z)BUFFER})
    (( $#words )) || return 1

    local cmd=$words[1] cur=$words[-1]
    [[ $cur == -* ]] || return 1
    (( ${+_comps[$cmd]} )) && return 1

    local flags
    flags=$(shelliq flags "$cmd" --raw 2>/dev/null) || return 1
    [[ -n $flags ]] || return 1

    local -a candidates
    candidates=(${(f)flags})
    compadd -Q -- $candidates
}

# Prepended, not overwritten: if the user's own .zshrc already set a `completer` list (most
# oh-my-zsh setups do), inserting `_shelliq` at the front keeps every existing completer's
# relative order intact — `_shelliq`'s own native-completion check (above) is what actually
# keeps it out of git/docker's way, not its position in this list. Always repositions to the
# front rather than only inserting when absent, so re-sourcing fixes a stale position instead
# of leaving one in place from an older version of this file.
typeset -a _shelliq_completers
zstyle -a ':completion:*' completer _shelliq_completers
(( $#_shelliq_completers )) || _shelliq_completers=(_complete)
_shelliq_completers=(_shelliq ${_shelliq_completers:#_shelliq})
zstyle ':completion:*' completer $_shelliq_completers
unset _shelliq_completers
