import io

from shelliq_training import progress


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_progress_tracks_stderr_tty(monkeypatch):
    monkeypatch.setattr(progress.sys, 'stderr', TtyBuffer())
    assert progress.interactive_progress()
    assert not progress.training_progress().disable
    assert not progress.evaluation_progress().disable


def test_progress_is_disabled_for_redirected_stderr(monkeypatch):
    monkeypatch.setattr(progress.sys, 'stderr', io.StringIO())
    assert not progress.interactive_progress()
    assert progress.training_progress().disable
    assert progress.evaluation_progress().disable
