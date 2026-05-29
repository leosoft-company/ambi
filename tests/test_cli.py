"""Cover the CLI bits that don't need a network — paths, version, init."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ambi.cli import paths
from ambi.cli.main import build_parser, cmd_init, cmd_version


def test_ambi_home_defaults_to_dot_ambi(monkeypatch, tmp_path):
    monkeypatch.delenv("AMBI_HOME", raising=False)
    with patch.object(Path, "home", return_value=tmp_path):
        assert paths.ambi_home() == tmp_path / ".ambi"


def test_ambi_home_respects_env(monkeypatch, tmp_path):
    override = tmp_path / "custom"
    monkeypatch.setenv("AMBI_HOME", str(override))
    assert paths.ambi_home() == override.resolve()


def test_ensure_tree_creates_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    paths.ensure_tree()
    assert paths.ambi_home().is_dir()
    assert paths.skills_dir().is_dir()
    assert paths.data_dir().is_dir()


def test_init_writes_template_and_skills(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    rc = cmd_init(None)
    assert rc == 0
    assert paths.env_file().exists()
    assert "GEMINI_API_KEY=" in paths.env_file().read_text()
    assert paths.system_md().exists()
    assert "You are ambi" in paths.system_md().read_text()
    assert (paths.skills_dir() / "time.md").exists()
    assert (paths.skills_dir() / "shell.md").exists()
    out = capsys.readouterr().out
    assert "Created:" in out


def test_init_does_not_overwrite_existing_system_md(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    paths.ensure_tree()
    paths.system_md().write_text("custom personality")
    cmd_init(None)
    assert paths.system_md().read_text() == "custom personality"


def test_load_system_prompt_uses_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    paths.ensure_tree()
    paths.system_md().write_text("only this matters")
    from ambi.cli.build import load_system_prompt
    assert load_system_prompt(with_hippocamp=False) == "only this matters"


def test_load_system_prompt_appends_hippocamp_addon(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    paths.ensure_tree()
    paths.system_md().write_text("base")
    from ambi.cli.build import load_system_prompt
    out = load_system_prompt(with_hippocamp=True)
    assert out.startswith("base")
    assert "Memory" in out
    assert "recall_memory" in out


def test_init_is_idempotent(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    cmd_init(None)
    capsys.readouterr()  # discard first output
    cmd_init(None)
    out = capsys.readouterr().out
    assert "Already initialized" in out


def test_init_does_not_overwrite_existing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBI_HOME", str(tmp_path / "h"))
    paths.ensure_tree()
    paths.env_file().write_text("GEMINI_API_KEY=existing-token\n")
    cmd_init(None)
    assert "existing-token" in paths.env_file().read_text()


def test_version_prints_version(capsys):
    rc = cmd_version(None)
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("ambi ")


def test_parser_knows_subcommands():
    parser = build_parser()
    args = parser.parse_args(["init"])
    assert args.command == "init"
    args = parser.parse_args(["run"])
    assert args.command == "run"
    args = parser.parse_args(["chat"])
    assert args.command == "chat"
    args = parser.parse_args(["version"])
    assert args.command == "version"
