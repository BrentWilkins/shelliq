from __future__ import annotations

from pathlib import Path

from shelliq_training.model_runtime import (
    CHECKPOINT_SHA256,
    INDEX_SHA256,
    _sha256,
    checkpoint_path,
    documentation_index_path,
    install_assets,
)


def test_packaged_runtime_assets_install_with_frozen_hashes(tmp_path: Path) -> None:
    index, checkpoint = install_assets(tmp_path)
    assert index == documentation_index_path(tmp_path)
    assert checkpoint == checkpoint_path(tmp_path)
    assert _sha256(index) == INDEX_SHA256
    assert _sha256(checkpoint) == CHECKPOINT_SHA256

    assert install_assets(tmp_path) == (index, checkpoint)
