"""Runtime workarounds that must happen before the ML stack loads.

Currently one, and it is a trap worth understanding before you touch an import
line anywhere in this package.

On Windows, several of our dependencies link their own OpenMP runtime. Once one
of them is in the process, torch's ``c10.dll`` can no longer initialise:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading ...\\torch\\lib\\c10.dll

Measured on this environment — torch second fails, torch first is fine:

    import sklearn ; import torch   -> WinError 1114
    import xgboost ; import torch   -> WinError 1114
    import shap    ; import torch   -> WinError 1114   (it imports sklearn)
    import numpy / pandas / scipy   -> fine either way

Two things make this hard to diagnose:

* The traceback surfaces at whatever imports torch *last*, which is almost
  always spaCy — it pulls torch in through ``thinc.compat`` whenever torch is
  installed. So a clash caused by ``import sklearn`` reads as a broken spaCy
  install, nowhere near the real cause.
* It is unconditional once torch is in the environment. It does not matter
  whether the caller wants the text channel; only which library got there
  first.

**The rule: any module that imports sklearn, xgboost or shap — at module level
or indirectly — must call ``preload_torch()`` before that import.** It is
already called by ``model_contract``, ``evaluation.metrics``,
``evaluation.cross_validate`` and ``scoring.train_xgboost``, which covers every
current entry point. Where torch is not installed it is a no-op, so the
stdlib-only test suite still runs on a bare checkout.
"""

from __future__ import annotations

_PRELOADED: bool | None = None


def preload_torch() -> bool:
    """Import torch ahead of xgboost. Returns True if torch is present.

    Safe to call repeatedly; the second call is free.
    """
    global _PRELOADED
    if _PRELOADED is not None:
        return _PRELOADED
    try:
        import torch  # noqa: F401  (imported for its DLL side effect)
        _PRELOADED = True
    except ImportError:
        # No text channel installed — nothing to order against.
        _PRELOADED = False
    return _PRELOADED
