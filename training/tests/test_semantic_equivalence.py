from shelliq_training.semantic_equivalence import canonicalize_semantic_document, semantic_documents_equivalent


def document(*arguments: str) -> dict[str, object]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': 'command'}, 'a': [{'s': value} for value in arguments]}]}],
    }


def test_split_and_clustered_short_flags_are_equivalent():
    assert semantic_documents_equivalent(document('-i', '-h', 'file'), document('-ih', 'file'))
    assert semantic_documents_equivalent(document('-R', '-n', 'TODO', 'src'), document('-Rn', 'TODO', 'src'))


def test_flag_order_and_missing_flags_remain_different():
    assert not semantic_documents_equivalent(document('-a', '-b'), document('-b', '-a'))
    assert not semantic_documents_equivalent(document('-R', '-n', 'TODO'), document('-R', 'TODO'))


def test_quotes_are_ignored_only_for_shell_safe_words():
    assert semantic_documents_equivalent(document("'hello.txt'"), document('hello.txt'))
    assert not semantic_documents_equivalent(document("'python.*server.py'"), document('python.*server.py'))
    assert not semantic_documents_equivalent(document("'two words'"), document('two words'))
    assert not semantic_documents_equivalent(document("'$HOME'"), document('$HOME'))


def test_canonicalization_does_not_mutate_input():
    original = document('-ih')
    canonical = canonicalize_semantic_document(original)
    assert original == document('-ih')
    assert canonical == document('-i', '-h')
