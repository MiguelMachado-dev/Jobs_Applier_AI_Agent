"""The folder AIHawk keeps everything in: ~/.aihawk, or the one AIHAWK_HOME names."""

from __future__ import annotations

from pathlib import Path

import pytest

from aihawk.config import HOME_ENV, default_home


def test_the_home_is_dot_aihawk_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOME_ENV, raising=False)
    assert default_home() == Path.home() / ".aihawk"


def test_the_variable_names_another_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "second"))
    assert default_home() == tmp_path / "second"


def test_a_blank_variable_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV, "  ")
    assert default_home() == Path.home() / ".aihawk"
