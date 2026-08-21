from scripts.run_teacher_cascade import attempt_accepted


def test_attempt_accepted_requires_an_empty_failure_list() -> None:
    assert attempt_accepted({'verification': {'failures': []}})
    assert not attempt_accepted({'verification': {'failures': ['reference-mismatch']}})
    assert not attempt_accepted({'verification': None})
