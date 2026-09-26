# SPDX-License-Identifier: Apache-2.0
"""``serve --served-model-alias``: one served model, every name its operator gives it.

goose serves a model under a node alias (``--served-model-name``) while OpenAI clients
name it by its Hugging Face id; before this flag the HF id got a 404 ("does not exist.
Available: <alias>"), measured against v0.14.3-lz.8's ``_validate_model_name`` with the
registry ``load_model`` builds for goose's argv. No model is loaded here.
"""

import asyncio

import pytest
from fastapi import HTTPException

from rapid_mlx.cli import build_parser
from rapid_mlx.config import get_config
from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from rapid_mlx.server import _entry_aliases
from rapid_mlx.service.helpers import _validate_model_name

PATH = "/Users/me/.goose/models/rapid-mlx/Qwen3.8-Flash-Next-4bit"
SERVED = "mihai-flash-qwen3.8-flash-next-4bit-mlx"
HF_ID = "rapid-mlx/Qwen3.8-Flash-Next-4bit"
NODE = "studio-qwen3.8-flash-next-4bit-mlx"


@pytest.fixture
def served(monkeypatch):
    registry = ModelRegistry()
    registry.add(
        ModelEntry(
            engine=object(),
            model_name=SERVED,
            model_path=PATH,
            aliases=_entry_aliases(SERVED, None, [HF_ID, NODE, SERVED]),
        ),
        is_default=True,
    )
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_registry", registry)
    monkeypatch.setattr(cfg, "model_name", SERVED)
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "model_path", PATH)
    return registry


def test_the_flag_is_repeatable_and_absent_by_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            PATH,
            "--served-model-name",
            SERVED,
            "--served-model-alias",
            HF_ID,
            "--served-model-alias",
            NODE,
        ]
    )
    assert args.served_model_alias == [HF_ID, NODE]
    assert parser.parse_args(["serve", PATH]).served_model_alias is None


def test_the_served_name_is_never_its_own_alias():
    assert _entry_aliases(SERVED, "qwen3.8-flash", [HF_ID, "", SERVED]) == {
        HF_ID,
        "qwen3.8-flash",
    }


def test_every_name_of_the_served_model_is_accepted(served):
    for name in (SERVED, HF_ID, NODE, PATH, "default"):
        _validate_model_name(name)
        assert served.get_entry(name).model_name == SERVED


def test_a_different_model_is_still_refused_by_name(served):
    with pytest.raises(HTTPException) as refusal:
        _validate_model_name("mihai-qwen3.8-27b-atlassian-q8-mlx")
    assert refusal.value.status_code == 404
    assert "mihai-qwen3.8-27b-atlassian-q8-mlx" in refusal.value.detail
    assert f"Available: {SERVED}, {HF_ID}, {NODE}" in refusal.value.detail


def test_models_lists_the_served_name_first_then_its_aliases(served):
    from rapid_mlx.routes.models import list_models

    listed = [model.id for model in asyncio.run(list_models()).data]
    assert listed == [SERVED, HF_ID, NODE]
