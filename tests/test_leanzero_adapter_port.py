"""Adapter forwarding must preserve v0.14's reviewed-runtime security overlay."""
from rapid_mlx.utils import tokenizer as loader


def test_reviewed_runtime_keeps_overlay_and_adapter(monkeypatch):
    monkeypatch.setattr(loader, '_register_vendored_archs', lambda: None)
    received = {}
    def fallback(model, **kwargs):
        received.update(kwargs)
        return 'model', 'tokenizer'
    monkeypatch.setattr(loader, '_load_with_tokenizer_fallback', fallback)
    overlay = {'model_file': None, 'auto_map': None}
    assert loader._load_model_with_fallback_impl('reviewed', model_config=overlay, adapter_path='/adapter') == ('model', 'tokenizer')
    assert received['model_config'] == overlay
    assert received['adapter_path'] == '/adapter'


def test_native_load_receives_neutralized_tokenizer_and_adapter(monkeypatch):
    import mlx_lm
    from rapid_mlx.models import gemma4_text
    class StopAtLoad(Exception):
        pass
    received = {}
    def load(model, **kwargs):
        received.update(kwargs)
        raise StopAtLoad()
    monkeypatch.setattr(mlx_lm, 'load', load)
    monkeypatch.setattr(loader, '_register_vendored_archs', lambda: None)
    monkeypatch.setattr(loader, '_neutralize_unbundled_template_types', lambda *_: {'safe': True})
    monkeypatch.setattr(loader, '_needs_tokenizer_fallback', lambda *_: False)
    monkeypatch.setattr(loader, '_is_vendored_arch_model', lambda *_: False)
    monkeypatch.setattr(gemma4_text, 'gemma4_load_plan', lambda *_: (None, False))
    import pytest
    with pytest.raises(StopAtLoad):
        loader._load_model_with_fallback_impl('native', {'unsafe': True}, adapter_path='/adapter')
    assert received == {'tokenizer_config': {'safe': True}, 'adapter_path': '/adapter'}
