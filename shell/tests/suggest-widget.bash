#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=shell/shelliq.bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/shelliq.bash"

TEST_CALLS_FILE=$(mktemp)
trap 'rm -f -- "$TEST_CALLS_FILE"' EXIT

reset_fixture() {
  READLINE_LINE=$1
  READLINE_POINT=${#READLINE_LINE}
  printf '0\n' > "$TEST_CALLS_FILE"
}

calls() {
  command cat -- "$TEST_CALLS_FILE"
}

shelliq() {
  if [[ ${1-} == explain ]]; then
    printf '%s\n' 'verified fixture explanation'
    return 0
  fi

  local call
  call=$(calls)
  (( call++ )) || true
  printf '%s\n' "$call" > "$TEST_CALLS_FILE"
  case ${TEST_SCENARIO}:${call} in
    ready:1)
      printf '%s\n' $'shelliq-widget-v1\tready\t66696e64202e202d747970652066'
      return 0
      ;;
    continuation:1|multi:1|cancel:1|changed:1)
      printf '%s\n' $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a64656d6f3a31\t57686963682066696c653f'
      return 1
      ;;
    continuation:2)
      [[ $* == *'--continue-from tldr:linux:demo:1 --answer notes.txt --zsh-widget'* ]]
      printf '%s\n' $'shelliq-widget-v1\tready\t64656d6f206e6f7465732e747874'
      return 0
      ;;
    multi:2)
      [[ $* == *'--answer first --zsh-widget'* ]]
      printf '%s\n' $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a64656d6f3a31\t5365636f6e642076616c75653f'
      return 1
      ;;
    multi:3)
      [[ $* == *'--answer first --answer second --zsh-widget'* ]]
      printf '%s\n' $'shelliq-widget-v1\tready\t64656d6f206669727374207365636f6e64'
      return 0
      ;;
    changed:2)
      printf '%s\n' $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a6f746865723a31\t5365636f6e642076616c75653f'
      return 1
      ;;
    malformed:1)
      printf '%s\n' 'not a protocol response'
      return 1
      ;;
  esac
  return 1
}

assert_equal() {
  if [[ $1 != "$2" ]]; then
    printf 'expected %q, got %q\n' "$2" "$1" >&2
    return 1
  fi
}

_shelliq_bash_decode_hex 636166c3a9
assert_equal "$REPLY" 'café'

reset_fixture 'grep -r needle .'
READLINE_POINT=4
original_point=$READLINE_POINT
_shelliq_bash_explain_widget
assert_equal "$READLINE_LINE" 'grep -r needle .'
assert_equal "$READLINE_POINT" "$original_point"

reset_fixture 'find regular files'
TEST_SCENARIO=ready
_shelliq_bash_suggest_widget
assert_equal "$READLINE_LINE" 'find . -type f'

reset_fixture 'run demo on a file'
TEST_SCENARIO=continuation
_shelliq_bash_suggest_widget < <(printf '%s\n' notes.txt)
assert_equal "$READLINE_LINE" 'demo notes.txt'
assert_equal "$(calls)" 2

reset_fixture 'run demo twice'
TEST_SCENARIO=multi
_shelliq_bash_suggest_widget < <(printf '%s\n' first second)
assert_equal "$READLINE_LINE" 'demo first second'
assert_equal "$(calls)" 3

reset_fixture 'keep this request'
TEST_SCENARIO=cancel
_shelliq_bash_suggest_widget < /dev/null
assert_equal "$READLINE_LINE" 'keep this request'

reset_fixture 'keep source pinned'
TEST_SCENARIO=changed
_shelliq_bash_suggest_widget < <(printf '%s\n' first)
assert_equal "$READLINE_LINE" 'keep source pinned'

reset_fixture 'keep malformed response'
TEST_SCENARIO=malformed
_shelliq_bash_suggest_widget
assert_equal "$READLINE_LINE" 'keep malformed response'

printf '%s\n' 'bash suggest widget tests passed'
