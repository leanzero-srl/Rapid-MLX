# SPDX-License-Identifier: Apache-2.0
"""One home for turning a prompt cache into state and back (mlx-lm 0.32 contract).

mlx-lm 0.32.0 changed the cache serialization contract under every caller:

* ``meta_state`` is gone from ``_BaseCache`` and every upstream class. The
  scalars it carried (``offset``, ``max_size``, ``_idx``, quantization bits...)
  now live INSIDE ``state``: ``KVCache.state`` is ``(keys, values, offset)``
  with the whole step-rounded buffers, ``RotatingKVCache.state`` adds
  ``keep / max_size / _idx``, ``ArraysCache.state`` is
  ``(cache, left_padding, lengths)`` and ``CacheList.state`` is
  ``[(child_state, child_class_name), ...]``.
* ``_BaseCache.from_state`` takes ONE argument, and ``CacheList``'s ``state``
  setter rebuilds its children through ``mlx_lm.models.cache`` globals only.

Rapid's own cache classes (DeepSeek-V4 pooling, Qwen4 QSA, the live quantized
batch cache, TurboQuant) still carry a ``meta_state`` and a two-argument
``from_state``. This module is the one place that knows both conventions, so
the persisted prefix cache (``memory_cache``) and the scheduler's mid-prefill
restore cannot drift apart:

* :func:`cache_meta` records a cache's own ``meta_state`` (and, for a
  ``CacheList``, its children's) — ``None`` for an upstream class whose
  scalars are already in ``state``;
* :func:`restore_cache` rebuilds a cache from ``(class name, state, meta)``,
  resolving Rapid-owned classes (also when nested in a ``CacheList``) through
  :func:`vendored_cache_registry` instead of patching mlx-lm's globals;
* :func:`save_prompt_cache` / :func:`load_prompt_cache` write and read a
  self-describing safetensors file whose leaf codec handles ``None``,
  zero-element arrays, Python scalars and strings, and that refuses a file
  written under the pre-0.32 layout (its states no longer fit the classes)
  with a ``ValueError`` the persistence layer counts as a skipped entry.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "CODEC_VERSION",
    "cache_meta",
    "load_prompt_cache",
    "restore_cache",
    "save_prompt_cache",
    "vendored_cache_registry",
]

CODEC_VERSION = "2"

_CODEC_KEY = "__rapid_mlx_cache_codec"
_CLASSES_KEY = "__rapid_mlx_cache_classes"
_META_KEY = "__rapid_mlx_cache_meta"
_LEAVES_KEY = "__rapid_mlx_cache_leaves"
_EMPTY_LAYERS_KEY = "__rapid_mlx_cache_empty_layers"
_VENDORED_KEY = "__rapid_mlx_vendored_cache_classes_v1"
_RESERVED = frozenset(
    {_CODEC_KEY, _CLASSES_KEY, _META_KEY, _LEAVES_KEY, _EMPTY_LAYERS_KEY, _VENDORED_KEY}
)


def vendored_cache_registry() -> dict[str, type]:
    """Cache classes owned by Rapid rather than ``mlx_lm``.

    Persisted prefix-cache restore and the scheduler's in-memory mid-prefill
    restore share this registry, so a vendored hybrid side-cache cannot
    restore after a restart but fail in the live scheduler path.
    """
    from rapid_mlx.models.deepseek_v4_cache import (
        BatchDeepseekV4PoolingCache,
        BatchPoolingCache,
        DeepseekV4PoolingCache,
        PoolingCache,
    )
    from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache, Qwen4ExpStateCache

    classes = (
        PoolingCache,
        BatchPoolingCache,
        DeepseekV4PoolingCache,
        BatchDeepseekV4PoolingCache,
        QSAIndexCache,
        Qwen4ExpStateCache,
    )
    return {cls.__name__: cls for cls in classes}


def _rapid_cache_classes() -> dict[str, type]:
    """Rapid-owned cache classes that mlx-lm's own module does not hold."""
    registry = dict(vendored_cache_registry())
    from rapid_mlx.quantized_batch_cache import QuantizedBatchKVCache
    from rapid_mlx.turboquant import TurboQuantKVCache

    registry.setdefault(QuantizedBatchKVCache.__name__, QuantizedBatchKVCache)
    registry.setdefault(TurboQuantKVCache.__name__, TurboQuantKVCache)
    return registry


def _is_cache_list(cache: Any) -> bool:
    return type(cache).__name__ == "CacheList" and hasattr(cache, "caches")


def cache_meta(cache: Any) -> Any:
    """The scalars a cache keeps OUTSIDE ``state``, as a JSON-able tree.

    ``None`` for an mlx-lm >= 0.32 class (everything is in ``state``); the
    class's own ``meta_state`` for a Rapid class that still keeps one; and
    ``{"children": [...]}`` for a ``CacheList``, so a Rapid class nested in it
    keeps its scalars too (0.32's ``CacheList.state`` carries only the
    children's ``state`` and class names).
    """
    if _is_cache_list(cache):
        return {"children": [cache_meta(child) for child in cache.caches]}
    try:
        meta = cache.meta_state
    except AttributeError:
        return None
    return _jsonable(meta)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _resolve_class(name: str, declared_vendored: frozenset[str] | None) -> type:
    from mlx_lm.models import cache as mlx_cache

    rapid = _rapid_cache_classes()
    upstream = getattr(mlx_cache, name, None)
    if name in rapid and (
        declared_vendored is None or name in declared_vendored or upstream is None
    ):
        return rapid[name]
    if isinstance(upstream, type):
        return upstream
    if name in rapid:
        return rapid[name]
    raise ValueError(f"unknown prompt-cache class {name!r}")


def restore_cache(
    cls_or_name: type | str,
    state: Any,
    meta: Any = None,
    *,
    declared_vendored: frozenset[str] | None = None,
) -> Any:
    """Rebuild one cache from its ``state`` and :func:`cache_meta` tree.

    ``declared_vendored`` (persisted files) names the classes the writer
    resolved to Rapid's registry; ``None`` resolves a Rapid class first.
    """
    name = cls_or_name if isinstance(cls_or_name, str) else cls_or_name.__name__
    if name == "CacheList":
        from mlx_lm.models.cache import CacheList

        children_meta = (meta or {}).get("children") if isinstance(meta, dict) else None
        children_state = list(state or ())
        if children_meta is None:
            children_meta = [None] * len(children_state)
        if len(children_meta) != len(children_state):
            raise ValueError(
                "CacheList state/metadata length mismatch: "
                f"state={len(children_state)}, metadata={len(children_meta)}"
            )
        children = []
        for (child_state, child_name), child_meta in zip(
            children_state, children_meta, strict=True
        ):
            children.append(
                restore_cache(
                    str(child_name),
                    child_state,
                    child_meta,
                    declared_vendored=declared_vendored,
                )
            )
        return CacheList(*children)
    cls = (
        cls_or_name
        if isinstance(cls_or_name, type)
        else _resolve_class(name, declared_vendored)
    )
    if meta is not None:
        return cls.from_state(state, meta)
    return cls.from_state(state)


# -- the persisted file -----------------------------------------------------


def _encode_leaf(key: str, value: Any, leaves: dict[str, Any], arrays: dict[str, Any]):
    import mlx.core as mx

    if value is None:
        leaves[key] = {"kind": "none"}
    elif isinstance(value, bool):
        leaves[key] = {"kind": "bool", "value": value}
    elif isinstance(value, int):
        leaves[key] = {"kind": "int", "value": value}
    elif isinstance(value, float):
        leaves[key] = {"kind": "float", "value": value}
    elif isinstance(value, str):
        leaves[key] = {"kind": "str", "value": value}
    elif isinstance(value, mx.array):
        if value.size == 0:
            # Kept out of the tensor block: its shape and dtype ride in the
            # leaf table, so a zero-length (B, H, 0, D) buffer comes back as
            # exactly that.
            leaves[key] = {
                "kind": "empty",
                "shape": list(value.shape),
                "dtype": str(value.dtype).rsplit(".", 1)[-1],
            }
        else:
            arrays[key] = value
    else:
        raise ValueError(
            f"prompt-cache state leaf {key!r} is not serializable: {type(value).__name__}"
        )


def _decode_leaf(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind")
    if kind == "none":
        return None
    if kind in ("bool", "int", "float", "str"):
        return spec["value"]
    if kind == "empty":
        import mlx.core as mx

        return mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=getattr(mx, spec["dtype"]))
    raise ValueError(f"unknown prompt-cache leaf kind {kind!r}")


def save_prompt_cache(path: str, cache: list[Any], metadata: dict[str, str]) -> None:
    """Write ``cache`` with the Rapid codec (see the module docstring)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    arrays: dict[str, Any] = {}
    leaves: dict[str, Any] = {}
    empty_layers: dict[str, str] = {}
    for index, layer in enumerate(cache):
        state = layer.state
        flat = tree_flatten(state)
        if not flat:
            # ``tree_flatten`` drops an empty container; remember its kind so
            # the layer comes back as itself, not as a neighbour's state.
            empty_layers[str(index)] = "dict" if isinstance(state, dict) else "list"
            continue
        for key, value in flat:
            _encode_leaf(f"{index}.{key}" if key else str(index), value, leaves, arrays)
    vendored = sorted(
        {
            name
            for name in _class_names(cache)
            if name in vendored_cache_registry()
        }
    )
    embedded = {k: v for k, v in (metadata or {}).items() if k not in _RESERVED}
    embedded[_CODEC_KEY] = CODEC_VERSION
    embedded[_CLASSES_KEY] = json.dumps([type(c).__name__ for c in cache])
    embedded[_META_KEY] = json.dumps([cache_meta(c) for c in cache])
    embedded[_LEAVES_KEY] = json.dumps(leaves, separators=(",", ":"))
    embedded[_EMPTY_LAYERS_KEY] = json.dumps(empty_layers)
    embedded[_VENDORED_KEY] = json.dumps(vendored)
    mx.save_safetensors(path, arrays, embedded)


def _class_names(cache: list[Any]):
    pending = list(cache)
    while pending:
        item = pending.pop()
        yield type(item).__name__
        if _is_cache_list(item):
            pending.extend(item.caches)


def load_prompt_cache(path: str, *, return_metadata: bool = False):
    """Read a file written by :func:`save_prompt_cache`."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    arrays, metadata = mx.load(path, return_metadata=True)
    if metadata.get(_CODEC_KEY) != CODEC_VERSION:
        raise ValueError(
            f"prompt cache {path!r} predates the mlx-lm 0.32 cache layout "
            f"(codec {metadata.get(_CODEC_KEY)!r}, need {CODEC_VERSION!r}); "
            "its states no longer fit the cache classes"
        )
    classes = json.loads(metadata[_CLASSES_KEY])
    metas = json.loads(metadata[_META_KEY])
    leaves = json.loads(metadata[_LEAVES_KEY])
    empty_layers = json.loads(metadata.get(_EMPTY_LAYERS_KEY, "{}"))
    declared_vendored = frozenset(json.loads(metadata.get(_VENDORED_KEY, "[]")))
    if len(metas) != len(classes):
        raise ValueError("prompt cache class/metadata length mismatch")

    per_layer: dict[int, list[tuple[str, Any]]] = {i: [] for i in range(len(classes))}
    for key, value in list(arrays.items()) + [
        (key, _decode_leaf(spec)) for key, spec in leaves.items()
    ]:
        head, _, rest = key.partition(".")
        per_layer[int(head)].append((rest, value))

    cache = []
    for index, (name, meta) in enumerate(zip(classes, metas, strict=True)):
        items = per_layer[index]
        if not items:
            kind = empty_layers.get(str(index), "list")
            state: Any = {} if kind == "dict" else []
        elif len(items) == 1 and items[0][0] == "":
            state = items[0][1]
        else:
            state = tree_unflatten(items)
        cache.append(
            restore_cache(name, state, meta, declared_vendored=declared_vendored)
        )
    if return_metadata:
        user = {k: v for k, v in metadata.items() if k not in _RESERVED}
        return cache, user
    return cache
