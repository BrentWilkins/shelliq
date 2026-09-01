# shelliq.bash — Bash Readline integration. Suggestions are opt-in and never
# auto-executed. Source this file from an interactive Bash configuration.

_shelliq_bash_widget_message() {
  printf '\n%s\n' "$1"
}

# C-x C-h explains the current command against the local index. It never writes
# READLINE_LINE or READLINE_POINT.
_shelliq_bash_explain_widget() {
  local request=${READLINE_LINE-} out
  if [[ -z $request ]]; then
    _shelliq_bash_widget_message 'shelliq: buffer is empty'
    return
  fi

  if out=$(shelliq explain -- "$request" 2>&1); then
    :
  fi
  _shelliq_bash_widget_message "$out"
}

_shelliq_bash_decode_hex() {
  local encoded=$1 decoded='' pair byte
  (( ${#encoded} % 2 == 0 )) || return 1
  [[ $encoded =~ ^([0-9a-f]{2})*$ ]] || return 1
  while [[ -n $encoded ]]; do
    pair=${encoded:0:2}
    [[ $pair != 00 ]] || return 1
    printf -v byte '%b' "\\x$pair" || return 1
    decoded+=$byte
    encoded=${encoded:2}
  done
  REPLY=$decoded
}

_shelliq_bash_complete() {
  COMPREPLY=()
  local command_name=${COMP_WORDS[0]-}
  local current=${COMP_WORDS[COMP_CWORD]-}
  local flags

  [[ -n $command_name && $current == -* ]] || return 1
  if ! flags=$(shelliq flags "$command_name" --raw 2>/dev/null); then
    return 1
  fi
  [[ -n $flags ]] || return 1

  mapfile -t COMPREPLY < <(compgen -W "$flags" -- "$current")
  (( ${#COMPREPLY[@]} > 0 ))
}

_shelliq_bash_register_completion() {
  # Bash invokes -D only when the command has no command-specific completion.
  # Preserve a user's existing default policy rather than replacing it.
  complete -p -D &>/dev/null && return
  complete -D -o bashdefault -o default -F _shelliq_bash_complete
}

# C-x C-g turns the current English Readline buffer into an editable suggestion.
# Missing values stay pinned to the initially reported documentation source.
# Cancellation and invalid responses leave the original request untouched.
_shelliq_bash_suggest_widget() {
  local request=${READLINE_LINE-}
  if [[ -z $request ]]; then
    _shelliq_bash_widget_message 'shelliq: type a request first'
    return
  fi

  local out response_status command source='' next_source question answer message
  local -a fields answer_args=()
  local steps=0 exit_status

  while (( steps <= 8 )); do
    if (( steps == 0 )); then
      if out=$(shelliq suggest --zsh-widget -- "$request" 2>/dev/null); then
        exit_status=0
      else
        exit_status=$?
      fi
    else
      if out=$(shelliq suggest --continue-from "$source" "${answer_args[@]}" --zsh-widget -- "$request" 2>/dev/null); then
        exit_status=0
      else
        exit_status=$?
      fi
    fi

    IFS=$'\t' read -r -a fields <<< "$out"
    if (( ${#fields[@]} < 2 )) || [[ ${fields[0]} != shelliq-widget-v1 ]]; then
      _shelliq_bash_widget_message 'shelliq: suggestion failed without a valid response'
      return
    fi
    response_status=${fields[1]}

    if [[ $response_status == ready ]]; then
      if (( exit_status != 0 || ${#fields[@]} != 3 )) || ! _shelliq_bash_decode_hex "${fields[2]}"; then
        _shelliq_bash_widget_message 'shelliq: inconsistent ready response rejected'
        return
      fi
      command=$REPLY
      if [[ -z $command || $command == *$'\n'* ]]; then
        _shelliq_bash_widget_message 'shelliq: invalid command response rejected'
        return
      fi
      READLINE_LINE=$command
      READLINE_POINT=${#READLINE_LINE}
      _shelliq_bash_widget_message 'shelliq: suggestion loaded; inspect it before pressing Enter'
      return
    fi

    if [[ $response_status != needs_input || $exit_status == 0 || ${#fields[@]} != 4 ]]; then
      message=$response_status
      if (( ${#fields[@]} == 3 )) && _shelliq_bash_decode_hex "${fields[2]}"; then
        message="$response_status: $REPLY"
      fi
      _shelliq_bash_widget_message "shelliq: ${message:-unknown}; request unchanged"
      return
    fi
    if (( steps == 8 )); then
      _shelliq_bash_widget_message 'shelliq: clarification limit reached; request unchanged'
      return
    fi

    _shelliq_bash_decode_hex "${fields[2]}" || {
      _shelliq_bash_widget_message 'shelliq: invalid continuation rejected; request unchanged'
      return
    }
    next_source=$REPLY
    _shelliq_bash_decode_hex "${fields[3]}" || {
      _shelliq_bash_widget_message 'shelliq: invalid clarification rejected; request unchanged'
      return
    }
    question=$REPLY
    if [[ -z $next_source || -z $question ]]; then
      _shelliq_bash_widget_message 'shelliq: incomplete clarification rejected; request unchanged'
      return
    fi
    if [[ -n $source && $next_source != "$source" ]]; then
      _shelliq_bash_widget_message 'shelliq: continuation source changed; request unchanged'
      return
    fi
    source=$next_source

    answer=''
    if ! IFS= read -r -p "$question " answer; then
      _shelliq_bash_widget_message 'shelliq: clarification cancelled; request unchanged'
      return
    fi
    if [[ -z $answer ]]; then
      _shelliq_bash_widget_message 'shelliq: empty clarification cancelled; request unchanged'
      return
    fi
    answer_args+=(--answer "$answer")
    (( ++steps ))
  done
}

if [[ $- == *i* ]]; then
  bind -x '"\C-x\C-h":_shelliq_bash_explain_widget'
  bind -x '"\C-x\C-g":_shelliq_bash_suggest_widget'
  _shelliq_bash_register_completion
fi
