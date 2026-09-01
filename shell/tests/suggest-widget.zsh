#!/usr/bin/env zsh

set -u

typeset -ga TEST_ZLE_MESSAGES
function zle {
    if [[ ${1-} == -M ]]; then
        TEST_ZLE_MESSAGES+=("${2-}")
    fi
}
function bindkey { : }

source "${0:A:h:h}/shelliq.zsh"
set -e

TEST_CALLS_FILE=$(mktemp)
trap 'rm -f -- "$TEST_CALLS_FILE"' EXIT

typeset -ga TEST_ANSWERS
integer TEST_ANSWER_INDEX=1
function read {
    local variable=${${@: -1}%%\?*}
    if (( TEST_ANSWER_INDEX > ${#TEST_ANSWERS} )); then
        return 1
    fi
    typeset -g "$variable=${TEST_ANSWERS[TEST_ANSWER_INDEX]}"
    (( TEST_ANSWER_INDEX++ ))
}

function reset_fixture {
    BUFFER=$1
    CURSOR=${#BUFFER}
    TEST_ANSWERS=()
    TEST_ANSWER_INDEX=1
    TEST_ZLE_MESSAGES=()
    print 0 >| "$TEST_CALLS_FILE"
}

function calls {
    < "$TEST_CALLS_FILE"
}

function shelliq {
    integer call=$(calls)
    (( call++ ))
    print $call >| "$TEST_CALLS_FILE"
    case ${TEST_SCENARIO}:${call} in
        ready:1)
            print -r -- $'shelliq-widget-v1\tready\t66696e64202e202d747970652066'
            return 0
            ;;
        continuation:1|multi:1|cancel:1|changed:1)
            print -r -- $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a64656d6f3a31\t57686963682066696c653f'
            return 1
            ;;
        continuation:2)
            [[ "$*" == *'--continue-from tldr:linux:demo:1 --answer notes.txt --zsh-widget'* ]]
            print -r -- $'shelliq-widget-v1\tready\t64656d6f206e6f7465732e747874'
            return 0
            ;;
        multi:2)
            [[ "$*" == *'--answer first --zsh-widget'* ]]
            print -r -- $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a64656d6f3a31\t5365636f6e642076616c75653f'
            return 1
            ;;
        multi:3)
            [[ "$*" == *'--answer first --answer second --zsh-widget'* ]]
            print -r -- $'shelliq-widget-v1\tready\t64656d6f206669727374207365636f6e64'
            return 0
            ;;
        changed:2)
            print -r -- $'shelliq-widget-v1\tneeds_input\t746c64723a6c696e75783a6f746865723a31\t5365636f6e642076616c75653f'
            return 1
            ;;
        malformed:1)
            print -r -- 'not json'
            return 1
            ;;
    esac
    return 1
}

function assert_equal {
    [[ $1 == $2 ]] || {
        print -u2 -r -- "expected ${(qqq)2}, got ${(qqq)1}"
        return 1
    }
}

_shelliq_decode_hex 636166c3a9
assert_equal "$REPLY" 'café'

reset_fixture 'find regular files'
TEST_SCENARIO=ready
_shelliq_suggest_widget
assert_equal "$BUFFER" 'find . -type f'

reset_fixture 'run demo on a file'
TEST_SCENARIO=continuation
TEST_ANSWERS=(notes.txt)
_shelliq_suggest_widget
assert_equal "$BUFFER" 'demo notes.txt'
assert_equal "$(calls)" 2

reset_fixture 'run demo twice'
TEST_SCENARIO=multi
TEST_ANSWERS=(first second)
_shelliq_suggest_widget
assert_equal "$BUFFER" 'demo first second'
assert_equal "$(calls)" 3

reset_fixture 'keep this request'
TEST_SCENARIO=cancel
_shelliq_suggest_widget
assert_equal "$BUFFER" 'keep this request'

reset_fixture 'keep source pinned'
TEST_SCENARIO=changed
TEST_ANSWERS=(first)
_shelliq_suggest_widget
assert_equal "$BUFFER" 'keep source pinned'

reset_fixture 'keep malformed response'
TEST_SCENARIO=malformed
_shelliq_suggest_widget
assert_equal "$BUFFER" 'keep malformed response'

print 'suggest widget tests passed'
