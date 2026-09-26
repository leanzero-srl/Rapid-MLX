"""The pipeline server answers to every name its operator gives the served model.

goose serves a split under a node alias while OpenAI clients name the model by its
Hugging Face id; lz-pipeline-qwen4.2 answered the HF id with 404 "model '<hf id>' is
not served here; this engine serves '<alias>'" (goose Q-131, measured on :8091). Rank
0's app is built over a stand-in tokenizer: no rank is launched and no model loaded.
"""

import argparse

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402

SERVED = "mihai-flash-qwen3.8-flash-next-4bit-mlx"
HF_ID = "rapid-mlx/Qwen3.8-Flash-Next-4bit"
NODE = "studio-qwen3.8-flash-next-4bit-mlx"
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


class _Tokenizer:
    # No template contract: the tools refusal that follows the model check proves a
    # request got past it without generating anything.
    chat_template = ""
    eos_token_ids = [0]


def _client(aliases=()) -> TestClient:
    state = serve._State(served=SERVED, aliases=aliases, context=512, max_batch=1)
    return TestClient(serve._build_app(state, _Tokenizer(), {0}))


def _chat(client: TestClient, model: str):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
        },
    )


def test_the_flag_is_repeatable_and_absent_by_default():
    parser = argparse.ArgumentParser()
    serve.add_arguments(parser)
    base = [
        "--model",
        "/m",
        "--served-model-name",
        SERVED,
        "--port",
        "1",
        "--context",
        "8",
    ]
    options = parser.parse_args(
        [*base, "--served-model-alias", HF_ID, "--served-model-alias", NODE]
    )
    assert options.served_model_alias == [HF_ID, NODE]
    assert parser.parse_args(base).served_model_alias is None


def test_models_lists_the_served_name_first_then_every_alias():
    listed = _client((HF_ID, NODE, SERVED)).get("/v1/models").json()["data"]
    assert [model["id"] for model in listed] == [SERVED, HF_ID, NODE]
    assert {model["context_window"] for model in listed} == {512}


def test_every_name_passes_the_model_check():
    client = _client((HF_ID, NODE))
    for name in (SERVED, HF_ID, NODE):
        answer = _chat(client, name)
        assert answer.status_code == 400, (name, answer.text)
        assert answer.json()["error"]["type"] == "tools_unsupported"


def test_a_different_model_is_refused_naming_what_is_served():
    answer = _chat(_client((HF_ID,)), "mihai-qwen3.8-27b-atlassian-q8-mlx")
    assert answer.status_code == 404
    assert answer.json()["error"] == {
        "message": "model 'mihai-qwen3.8-27b-atlassian-q8-mlx' is not served here; this "
        f"engine serves '{SERVED}' (also answering to '{HF_ID}')",
        "type": "model_not_found",
    }


def test_without_aliases_only_the_served_name_answers():
    client = _client()
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [SERVED]
    answer = _chat(client, HF_ID)
    assert answer.status_code == 404
    assert answer.json()["error"]["message"] == (
        f"model '{HF_ID}' is not served here; this engine serves '{SERVED}'"
    )
