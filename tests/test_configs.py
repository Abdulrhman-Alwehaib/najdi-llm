from pathlib import Path

import pytest

from najdi import cli, settings
from najdi.science.config import Config as ScienceConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

FILES = {
    "science": "science.yaml",
    "chat-data": "chat-data.yaml",
    "clean": "clean.yaml",
    "mix": "mix.yaml",
    "compare": "compare.yaml",
    "train": "train.yaml",
    "serve": "serve.yaml",
}


@pytest.mark.parametrize("command", list(FILES))
def test_every_config_loads(command):
    cls = cli._resolve(cli.COMMANDS[command][0])
    cfg = settings.load(cls, CONFIGS / FILES[command])
    assert isinstance(cfg, cls)


def test_medical_is_a_science_config():
    cfg = settings.load(ScienceConfig, CONFIGS / "medical.yaml")
    assert cfg.domain.name == "medical"
    assert set(cfg.domain.topics) == {"medical"}
    assert cfg.domain.extra_rules.startswith("- ")
    assert cfg.id_prefix == "med" and cfg.science_frac == 1.0


def test_overrides_and_unknown_keys(tmp_path):
    cfg = settings.load(
        ScienceConfig, CONFIGS / "science.yaml", ["pilot=false", "wiki.label_chunk=7", "engine.max_num_seqs=8"]
    )
    assert cfg.pilot is False and cfg.wiki.label_chunk == 7 and cfg.engine.max_num_seqs == 8
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        settings.load(ScienceConfig, bad)


def test_cli_requires_a_config():
    with pytest.raises(SystemExit):
        cli.main(["train"])
