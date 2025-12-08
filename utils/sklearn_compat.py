"""
Compatibility helpers for scikit-learn deprecations.
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any


def patch_force_all_finite_alias() -> None:
    """
    Map deprecated ``force_all_finite`` keyword to the new
    ``ensure_all_finite`` name to keep third-party libraries (e.g., hdbscan)
    quiet on scikit-learn >= 1.6 and future-proof once the old name is
    removed in 1.8.
    """
    try:
        from sklearn.utils import validation as validation_utils
    except Exception:
        return

    check_array = validation_utils.check_array

    # Skip if the new parameter is not present (older sklearn) or already patched.
    if "ensure_all_finite" not in inspect.signature(check_array).parameters:
        return
    if getattr(validation_utils, "_force_all_finite_patched", False):
        return

    @wraps(check_array)
    def _check_array_compat(*args: Any, **kwargs: Any):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return check_array(*args, **kwargs)

    # Patch sklearn and mark so we don't wrap multiple times.
    validation_utils.check_array = _check_array_compat
    validation_utils._force_all_finite_patched = True

    # Patch the copy inside hdbscan so it uses the wrapped version too.
    try:
        import hdbscan.hdbscan_ as hdbscan_module

        hdbscan_module.check_array = _check_array_compat
    except Exception:
        # Best effort; if hdbscan is missing we still improve sklearn usage.
        pass
