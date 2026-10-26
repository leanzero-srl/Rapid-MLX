# SPDX-License-Identifier: Apache-2.0
"""mlx-lm model architectures the pinned mlx-lm release does not ship yet.

``import rapid_mlx.mlx_lm_models`` registers every module listed in ``VENDORED``
as ``sys.modules["mlx_lm.models.<model_type>"]``, so mlx-lm's own loaders
(``mlx_lm.utils.load`` / ``load_model`` / ``sharded_load``, which resolve
``importlib.import_module(f"mlx_lm.models.{model_type}")``) find it with no
other change. The import is cheap: it pulls in mlx / mlx-lm model helpers only,
never the server, engine or scheduler, and registering twice is a no-op.

A native module wins: when the installed mlx-lm already has
``mlx_lm/models/<model_type>.py`` nothing is registered for that type, so a
future pin bump never runs a stale vendored copy. ``REGISTERED`` records what
happened per type ("vendored" or "native").

The single engine calls ``register()`` from
``rapid_mlx.utils.tokenizer._register_vendored_archs`` on every model load.

Vendored modules (provenance and drop condition in each file's header):
  mimo_v2  ml-explore/mlx-lm PR #1219 @ e6552fdeae02 (Xiaomi MiMo-V2.5 / V2.6-Flash)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

VENDORED: tuple[str, ...] = ("mimo_v2",)
REGISTERED: dict[str, str] = {}


def register() -> dict[str, str]:
    """Install the vendored modules under ``mlx_lm.models.*``. Idempotent."""
    for model_type in VENDORED:
        if model_type in REGISTERED:
            continue
        name = f"mlx_lm.models.{model_type}"
        if name in sys.modules:
            mod = sys.modules[name]
            REGISTERED[model_type] = (
                "vendored" if mod.__name__.startswith(__name__) else "native"
            )
            continue
        try:
            native = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            native = None
        if native is not None:
            REGISTERED[model_type] = "native"
            continue
        mod = importlib.import_module(f"{__name__}.{model_type}")
        sys.modules.setdefault(name, mod)
        REGISTERED[model_type] = "vendored"
    return dict(REGISTERED)


register()
